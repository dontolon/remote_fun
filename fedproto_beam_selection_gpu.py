import os
import glob
import json
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Optional
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


NUM_CLASSES = 64
PWR_COLS = [f"pwr_{i}" for i in range(1, NUM_CLASSES + 1)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(device: torch.device, deterministic: bool = False) -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def get_autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class BeamRegressor(nn.Module):
    """
    MLP that predicts the full normalized beam power vector.
    Output logits are converted to a distribution with softmax.
    """

    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        nodes_per_layer: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        layers = []
        in_dim = num_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, nodes_per_layer))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = nodes_per_layer

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        pred_dist = torch.softmax(logits, dim=1)
        return logits, pred_dist


@dataclass
class ClientData:
    client_id: str
    train_loader: DataLoader
    num_samples: int


class TestDatasetBundle:
    def __init__(
        self,
        x_test: np.ndarray,
        y_test_label: np.ndarray,
        y_test_norm: np.ndarray,
        power_sums: np.ndarray,
        client_ids: list[str],
    ):
        self.x_test = x_test
        self.y_test_label = y_test_label
        self.y_test_norm = y_test_norm
        self.power_sums = power_sums
        self.client_ids = client_ids


def load_client_arrays(csv_path: str):
    df = pd.read_csv(csv_path)

    if "gps_lat" not in df.columns or "gps_long" not in df.columns:
        raise ValueError(f"{csv_path} must contain gps_lat and gps_long columns")

    missing_cols = [col for col in PWR_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing power columns in {csv_path}: {missing_cols}")

    x = df[["gps_lat", "gps_long"]].values.astype(np.float32)
    y_vec = df[PWR_COLS].values.astype(np.float32)

    if "power_sum" in df.columns:
        power_sum = df["power_sum"].values.astype(np.float32)
    else:
        power_sum = np.sum(y_vec, axis=1).astype(np.float32)

    y_norm = y_vec / (power_sum[:, None] + 1e-8)
    y_label = np.argmax(y_norm, axis=1).astype(np.int64)

    if "client_id" in df.columns:
        client_ids = df["client_id"].astype(str).tolist()
    else:
        client_ids = [os.path.basename(csv_path)] * len(df)

    return x, y_label, y_norm, power_sum, client_ids


def compute_global_feature_stats(train_folder: str) -> tuple[np.ndarray, np.ndarray]:
    train_paths = sorted(glob.glob(os.path.join(train_folder, "*.csv")))
    if not train_paths:
        raise FileNotFoundError(f"No train CSV files found in {train_folder}")

    total_count = 0
    sum_x = np.zeros(2, dtype=np.float64)
    sum_x2 = np.zeros(2, dtype=np.float64)

    for csv_path in train_paths:
        df = pd.read_csv(csv_path, usecols=["gps_lat", "gps_long"])
        x = df[["gps_lat", "gps_long"]].values.astype(np.float64)
        total_count += x.shape[0]
        sum_x += x.sum(axis=0)
        sum_x2 += (x ** 2).sum(axis=0)

    mean = sum_x / max(total_count, 1)
    var = sum_x2 / max(total_count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))

    return mean.astype(np.float32), std.astype(np.float32)


def normalize_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / np.clip(std[None, :], 1e-8, None)).astype(np.float32)


def make_dataloader(
    x: np.ndarray,
    y_norm: np.ndarray,
    y_label: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y_norm),
        torch.from_numpy(y_label),
    )

    persistent_workers = bool(num_workers > 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )


def select_participating_clients(
    num_clients: int,
    fraction_fit: float,
    seed: int,
    round_idx: int,
) -> list[int]:
    if fraction_fit >= 1.0:
        return list(range(num_clients))

    rng = random.Random(seed + round_idx)
    num_selected = max(1, int(np.ceil(fraction_fit * num_clients)))
    return sorted(rng.sample(range(num_clients), num_selected))


def weighted_average_state_dicts(
    local_state_dicts: list[dict[str, torch.Tensor]],
    local_num_samples: list[int],
) -> dict[str, torch.Tensor]:
    if not local_state_dicts:
        raise ValueError("No local state dicts provided for aggregation")

    total_samples = float(sum(local_num_samples))
    avg_state = {}

    for key in local_state_dicts[0].keys():
        weighted_sum = None
        for state_dict, num_samples in zip(local_state_dicts, local_num_samples):
            tensor = state_dict[key].float()
            weight = float(num_samples) / total_samples
            contrib = tensor * weight
            weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib
        avg_state[key] = weighted_sum

    return avg_state


def weighted_mean(values: list[float], weights: list[int]) -> float:
    if len(values) == 0:
        return float("nan")
    values_arr = np.asarray(values, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values_arr * weights_arr) / np.sum(weights_arr))


def compute_power_loss_db(selected_indices: list[int], true_vec_norm: np.ndarray, power_sum: float) -> float:
    if len(selected_indices) == 0:
        return np.nan

    true_vec = true_vec_norm * power_sum
    noise = np.min(true_vec) / 2.0
    measured_power = np.max(true_vec[np.asarray(selected_indices, dtype=np.int64)])
    true_max = np.max(true_vec)

    if measured_power - noise <= 0:
        return np.nan

    return float(10.0 * np.log10((true_max - noise) / (measured_power - noise)))


def prediction_loss(
    pred_dist: torch.Tensor,
    target_dist: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred_dist, target_dist, reduction="mean")

    if loss_name == "smoothl1":
        return F.smooth_l1_loss(pred_dist, target_dist, reduction="mean")

    if loss_name == "kl":
        log_pred = torch.log(pred_dist.clamp_min(1e-8))
        return F.kl_div(log_pred, target_dist, reduction="batchmean")

    raise ValueError(f"Unsupported loss: {loss_name}")


class BeamClient:
    def __init__(
        self,
        client_data: ClientData,
        device: torch.device,
        lr: float,
        decay_l2: float,
        local_epochs: int,
        use_amp: bool,
        fl_method: str,
        prox_mu: float,
        local_optimizer: str,
    ):
        self.client_data = client_data
        self.device = device
        self.lr = lr
        self.decay_l2 = decay_l2
        self.local_epochs = local_epochs
        self.use_amp = use_amp and device.type == "cuda"
        self.fl_method = fl_method
        self.prox_mu = prox_mu
        self.local_optimizer = local_optimizer.lower()

    def _build_local_model(self, global_model: nn.Module) -> nn.Module:
        return copy.deepcopy(global_model).to(self.device)

    def _make_optimizer(self, model: nn.Module):
        if self.local_optimizer == "adam":
            return optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.decay_l2)
        if self.local_optimizer == "adamw":
            return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.decay_l2)
        if self.local_optimizer == "sgd":
            return optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.decay_l2)
        raise ValueError(f"Unsupported optimizer: {self.local_optimizer}")

    def _proximal_term(
        self,
        local_model: nn.Module,
        global_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        prox = torch.zeros((), device=self.device)
        for name, param in local_model.named_parameters():
            prox = prox + torch.sum((param - global_params[name]) ** 2)
        return 0.5 * self.prox_mu * prox

    def local_train(
        self,
        global_model: nn.Module,
        loss_name: str,
    ) -> dict:
        model = self._build_local_model(global_model)
        model.train()

        optimizer = self._make_optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        if self.fl_method == "fedprox":
            global_params = {
                name: param.detach().clone().to(self.device)
                for name, param in global_model.named_parameters()
            }
        else:
            global_params = None

        total_loss = 0.0
        total_main_loss = 0.0
        total_prox_loss = 0.0
        total_samples = 0

        for _ in range(self.local_epochs):
            for x_batch, y_dist_batch, _ in self.client_data.train_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_dist_batch = y_dist_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with get_autocast_context(self.device, self.use_amp):
                    _, pred_dist = model(x_batch)
                    main_loss = prediction_loss(pred_dist, y_dist_batch, loss_name)

                    if self.fl_method == "fedprox":
                        prox_loss = self._proximal_term(model, global_params)
                    else:
                        prox_loss = torch.zeros((), device=self.device)

                    loss = main_loss + prox_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = x_batch.size(0)
                total_loss += float(loss.detach().item()) * batch_size
                total_main_loss += float(main_loss.detach().item()) * batch_size
                total_prox_loss += float(prox_loss.detach().item()) * batch_size
                total_samples += batch_size

        return {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "loss": total_loss / max(total_samples, 1),
            "main_loss": total_main_loss / max(total_samples, 1),
            "prox_loss": total_prox_loss / max(total_samples, 1),
            "num_samples": self.client_data.num_samples,
        }


def build_clients(
    args: argparse.Namespace,
    device: torch.device,
    x_mean: Optional[np.ndarray],
    x_std: Optional[np.ndarray],
) -> list[BeamClient]:
    train_paths = sorted(glob.glob(os.path.join(args.train_folder, "*.csv")))
    if not train_paths:
        raise FileNotFoundError(f"No train CSV files found in {args.train_folder}")

    clients = []
    for train_csv in train_paths:
        x_train, y_train_label, y_train_norm, _, train_client_ids = load_client_arrays(train_csv)

        if args.standardize_x:
            x_train = normalize_features(x_train, x_mean, x_std)

        client_id = str(train_client_ids[0]) if len(train_client_ids) > 0 else f"client_{len(clients)}"

        train_loader = make_dataloader(
            x_train,
            y_train_norm,
            y_train_label,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        client_data = ClientData(
            client_id=client_id,
            train_loader=train_loader,
            num_samples=len(y_train_label),
        )

        client = BeamClient(
            client_data=client_data,
            device=device,
            lr=args.lr,
            decay_l2=args.decay_l2,
            local_epochs=args.local_epochs,
            use_amp=args.use_amp,
            fl_method=args.fl_method,
            prox_mu=args.prox_mu,
            local_optimizer=args.local_optimizer,
        )
        clients.append(client)

    return clients


def load_all_test_data(
    test_folder: str,
    standardize_x: bool,
    x_mean: Optional[np.ndarray],
    x_std: Optional[np.ndarray],
) -> TestDatasetBundle:
    test_paths = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
    if not test_paths:
        raise FileNotFoundError(f"No test CSV files found in {test_folder}")

    xs, ys_label, y_norms, power_sums, client_ids = [], [], [], [], []

    for test_csv in test_paths:
        x_test, y_test_label, y_test_norm, ps_test, ids_test = load_client_arrays(test_csv)

        if standardize_x:
            x_test = normalize_features(x_test, x_mean, x_std)

        xs.append(x_test)
        ys_label.append(y_test_label)
        y_norms.append(y_test_norm)
        power_sums.append(ps_test)
        client_ids.extend(ids_test)

    return TestDatasetBundle(
        x_test=np.vstack(xs),
        y_test_label=np.concatenate(ys_label),
        y_test_norm=np.vstack(y_norms),
        power_sums=np.concatenate(power_sums),
        client_ids=client_ids,
    )


def federated_train(
    args: argparse.Namespace,
    clients: list[BeamClient],
    global_model: nn.Module,
) -> nn.Module:
    for round_idx in range(1, args.rounds + 1):
        selected_clients = select_participating_clients(
            num_clients=len(clients),
            fraction_fit=args.fraction_fit,
            seed=args.seed,
            round_idx=round_idx,
        )

        local_state_dicts = []
        local_num_samples = []
        round_losses = []
        round_main_losses = []
        round_prox_losses = []

        for client_idx in selected_clients:
            result = clients[client_idx].local_train(
                global_model=global_model,
                loss_name=args.loss,
            )

            local_state_dicts.append(result["state_dict"])
            local_num_samples.append(result["num_samples"])
            round_losses.append(result["loss"])
            round_main_losses.append(result["main_loss"])
            round_prox_losses.append(result["prox_loss"])

        new_global_state = weighted_average_state_dicts(local_state_dicts, local_num_samples)
        global_model.load_state_dict(new_global_state, strict=True)

        print(
            f"Round {round_idx:03d}/{args.rounds} | "
            f"method={args.fl_method} | "
            f"clients={len(selected_clients)} | "
            f"loss={weighted_mean(round_losses, local_num_samples):.6f} | "
            f"main={weighted_mean(round_main_losses, local_num_samples):.6f} | "
            f"prox={weighted_mean(round_prox_losses, local_num_samples):.6f}"
        )

    return global_model


def evaluate_and_save(
    args: argparse.Namespace,
    global_model: nn.Module,
    test_bundle: TestDatasetBundle,
    device: torch.device,
    x_mean: Optional[np.ndarray],
    x_std: Optional[np.ndarray],
) -> None:
    global_model.eval()
    global_model.to(device)

    test_loader = make_dataloader(
        test_bundle.x_test,
        test_bundle.y_test_norm,
        test_bundle.y_test_label,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    metrics_rows = []
    max_top_k = args.max_top_k
    sample_offset = 0

    with torch.no_grad():
        for x_batch, _, y_label_batch in test_loader:
            batch_size = x_batch.size(0)
            x_batch = x_batch.to(device, non_blocking=True)

            with get_autocast_context(device, args.use_amp and device.type == "cuda"):
                _, pred_dist = global_model(x_batch)

            pred_dist_np = pred_dist.float().cpu().numpy()
            topk_indices = np.argsort(-pred_dist_np, axis=1)[:, :max_top_k]
            y_true_batch = y_label_batch.numpy()

            for i in range(batch_size):
                global_idx = sample_offset + i
                selected_topk = topk_indices[i].tolist()
                selected_for_power = selected_topk[: args.power_loss_k]
                true_best = int(y_true_batch[i])

                power_loss_db = compute_power_loss_db(
                    selected_for_power,
                    test_bundle.y_test_norm[global_idx],
                    float(test_bundle.power_sums[global_idx]),
                )

                row = {
                    "client_id": test_bundle.client_ids[global_idx],
                    "selected_indices_topk": json.dumps(selected_topk),
                    "true_best": true_best,
                    "predicted_best": int(selected_topk[0]),
                    "best_power": float(
                        test_bundle.y_test_norm[global_idx, true_best] * test_bundle.power_sums[global_idx]
                    ),
                    f"power_loss_db_top{args.power_loss_k}": power_loss_db,
                }

                for k in range(1, max_top_k + 1):
                    row[f"hit@{k}"] = int(true_best in selected_topk[:k])

                metrics_rows.append(row)

            sample_offset += batch_size

    metrics_df = pd.DataFrame(metrics_rows)
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "num_samples": int(len(metrics_df)),
        "num_train_clients": int(len(glob.glob(os.path.join(args.train_folder, "*.csv")))),
        "fl_method": args.fl_method,
        "loss": args.loss,
        "prox_mu": float(args.prox_mu),
        "local_optimizer": args.local_optimizer,
        "standardize_x": bool(args.standardize_x),
        "power_loss_k": int(args.power_loss_k),
    }

    for k in range(1, max_top_k + 1):
        summary[f"top_{k}_accuracy"] = float(metrics_df[f"hit@{k}"].mean())

    summary[f"mean_power_loss_db_top{args.power_loss_k}"] = float(
        metrics_df[f"power_loss_db_top{args.power_loss_k}"].mean()
    )

    summary_df = pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()])
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    run_config = vars(args).copy()
    if x_mean is not None:
        run_config["x_mean"] = [float(v) for v in x_mean]
    if x_std is not None:
        run_config["x_std"] = [float(v) for v in x_std]

    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    model_path = os.path.join(args.output_dir, "global_model.pt")
    torch.save(global_model.state_dict(), model_path)

    print(f"\nPer-sample metrics saved to {metrics_path}")
    print(f"Summary saved to {summary_path}")
    print(f"Run config saved to {config_path}")
    print(f"Global model saved to {model_path}")

    print("\nTop-k accuracy summary:")
    for k in range(1, max_top_k + 1):
        print(f"Top-{k} accuracy: {summary[f'top_{k}_accuracy']:.4f}")
    print(f"Mean power loss (top {args.power_loss_k}): {summary[f'mean_power_loss_db_top{args.power_loss_k}']:.4f} dB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated regression of full normalized beam vector (FedAvg / FedProx)"
    )

    parser.add_argument("--train_folder", type=str, default="deepSense/train_sequences_scen1")
    parser.add_argument("--test_folder", type=str, default="deepSense/test_sequences_scen1")
    parser.add_argument("--output_dir", type=str, default="results_regression")

    parser.add_argument("--nodes_per_layer", type=int, default=128)
    parser.add_argument("--layers", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--decay_l2", type=float, default=1e-5)

    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--fraction_fit", type=float, default=1.0)

    parser.add_argument("--fl_method", type=str, default="fedprox", choices=["fedavg", "fedprox"])
    parser.add_argument("--prox_mu", type=float, default=1e-3)

    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1", "kl"])
    parser.add_argument("--local_optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])

    parser.add_argument("--max_top_k", type=int, default=10)
    parser.add_argument("--power_loss_k", type=int, default=10)

    parser.add_argument("--standardize_x", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_top_k < 1 or args.max_top_k > NUM_CLASSES:
        raise ValueError(f"max_top_k must be in [1, {NUM_CLASSES}]")
    if args.power_loss_k < 1 or args.power_loss_k > args.max_top_k:
        raise ValueError("power_loss_k must be in [1, max_top_k]")
    if args.fraction_fit <= 0 or args.fraction_fit > 1.0:
        raise ValueError("fraction_fit must be in (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configure_torch(device, deterministic=args.deterministic)
    set_seed(args.seed)

    print(f"Using device: {device}")
    print(f"Train folder: {args.train_folder}")
    print(f"Test folder:  {args.test_folder}")
    print(f"Federated method: {args.fl_method}")
    print(f"Loss: {args.loss}")

    if args.standardize_x:
        x_mean, x_std = compute_global_feature_stats(args.train_folder)
        print(f"Feature standardization enabled | mean={x_mean.tolist()} | std={x_std.tolist()}")
    else:
        x_mean, x_std = None, None
        print("Feature standardization disabled")

    global_model = BeamRegressor(
        num_features=2,
        num_outputs=NUM_CLASSES,
        nodes_per_layer=args.nodes_per_layer,
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    clients = build_clients(args, device, x_mean, x_std)
    print(f"Built {len(clients)} training clients")

    test_bundle = load_all_test_data(
        test_folder=args.test_folder,
        standardize_x=args.standardize_x,
        x_mean=x_mean,
        x_std=x_std,
    )
    print(f"Loaded {len(test_bundle.y_test_label)} test samples from {args.test_folder}")

    global_model = federated_train(args, clients, global_model)
    evaluate_and_save(args, global_model, test_bundle, device, x_mean, x_std)


if __name__ == "__main__":
    main()
