import os
import glob
import json
import copy
import random
import argparse
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Optional
import time
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


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
        y_test_db: np.ndarray,
        y_test_norm_linear: np.ndarray,
        y_test_linear: np.ndarray,
        power_sums: np.ndarray,
        client_ids: list[str],
    ):
        self.x_test = x_test
        self.y_test_label = y_test_label
        self.y_test_db = y_test_db
        self.y_test_norm_linear = y_test_norm_linear
        self.y_test_linear = y_test_linear
        self.power_sums = power_sums
        self.client_ids = client_ids


def load_client_arrays(
    csv_path: str,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
    assume_powers_are_linear: bool = True,
):
    df = pd.read_csv(csv_path)

    if "gps_lat" not in df.columns or "gps_long" not in df.columns:
        raise ValueError(f"{csv_path} must contain gps_lat and gps_long columns")

    missing_cols = [col for col in PWR_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing power columns in {csv_path}: {missing_cols}")

    x = df[["gps_lat", "gps_long"]].values.astype(np.float32)
    y_linear = df[PWR_COLS].values.astype(np.float32)

    if "power_sum" in df.columns:
        power_sum = df["power_sum"].values.astype(np.float32)
    else:
        power_sum = np.sum(y_linear, axis=1).astype(np.float32)

    if assume_powers_are_linear:
        eps = 1e-12
        y_db = 10.0 * np.log10(np.clip(y_linear, eps, None)).astype(np.float32)
    else:
        raise ValueError("This script assumes pwr_* columns are linear powers.")

    if target_mean is not None and target_std is not None:
        y_db = ((y_db - target_mean) / max(target_std, 1e-8)).astype(np.float32)

    y_norm_linear = y_linear / (power_sum[:, None] + 1e-8)
    y_label = np.argmax(y_linear, axis=1).astype(np.int64)

    if "client_id" in df.columns:
        client_ids = df["client_id"].astype(str).tolist()
    else:
        client_ids = [os.path.basename(csv_path)] * len(df)

    return x, y_label, y_db, y_norm_linear, y_linear, power_sum, client_ids


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


def compute_global_target_stats(train_folder: str) -> tuple[float, float]:
    train_paths = sorted(glob.glob(os.path.join(train_folder, "*.csv")))
    if not train_paths:
        raise FileNotFoundError(f"No train CSV files found in {train_folder}")

    total_count = 0
    sum_y = 0.0
    sum_y2 = 0.0

    for csv_path in train_paths:
        df = pd.read_csv(csv_path, usecols=PWR_COLS)
        y_linear = df[PWR_COLS].values.astype(np.float32)
        y_db = 10.0 * np.log10(np.clip(y_linear, 1e-12, None))
        total_count += y_db.size
        sum_y += float(y_db.sum())
        sum_y2 += float((y_db ** 2).sum())

    mean = sum_y / max(total_count, 1)
    var = sum_y2 / max(total_count, 1) - mean ** 2
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), std


def normalize_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / np.clip(std[None, :], 1e-8, None)).astype(np.float32)


def make_dataloader(
    x: np.ndarray,
    y_target: np.ndarray,
    y_label: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y_target),
        torch.from_numpy(y_label),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
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


def prediction_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target, reduction="mean")
    if loss_name == "smoothl1":
        return F.smooth_l1_loss(pred, target, reduction="mean")
    raise ValueError(f"Unsupported loss: {loss_name}")


def clone_state_dict(state_dict: dict[str, torch.Tensor], device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        t = v.detach().clone()
        if device is not None:
            t = t.to(device)
        out[k] = t
    return out


def init_control_variates_from_model(model: nn.Module, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
    ctrl = {}
    for name, param in model.state_dict().items():
        t = torch.zeros_like(param, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        ctrl[name] = t
    return ctrl


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
        grad_clip_norm: float,
    ):
        self.client_data = client_data
        self.device = device
        self.lr = lr
        self.decay_l2 = decay_l2
        self.local_epochs = local_epochs
        self.use_amp = use_amp and device.type == "cuda"
        self.fl_method = fl_method
        self.grad_clip_norm = grad_clip_norm
        self.c_local = None

    def _build_local_model(self, global_model: nn.Module) -> nn.Module:
        return copy.deepcopy(global_model).to(self.device)

    def _make_optimizer(self, model: nn.Module):
        return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.decay_l2)

    def _count_local_steps(self) -> int:
        num_batches = len(self.client_data.train_loader)
        return self.local_epochs * num_batches

    def local_train(
        self,
        global_model: nn.Module,
        global_control: Optional[dict[str, torch.Tensor]],
        loss_name: str,
    ) -> dict:
        model = self._build_local_model(global_model)
        model.train()

        optimizer = self._make_optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        global_state = clone_state_dict(global_model.state_dict(), device=self.device)

        if self.fl_method == "scaffold":
            if self.c_local is None:
                self.c_local = init_control_variates_from_model(global_model, device=self.device)
            c_global = clone_state_dict(global_control, device=self.device)
        else:
            c_global = None

        total_loss = 0.0
        total_samples = 0
        total_steps = 0

        for _ in range(self.local_epochs):
            for x_batch, y_target_batch, _ in self.client_data.train_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_target_batch = y_target_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with get_autocast_context(self.device, self.use_amp):
                    pred = model(x_batch)
                    loss = prediction_loss(pred, y_target_batch, loss_name)

                scaler.scale(loss).backward()

                if self.fl_method == "scaffold":
                    scaler.unscale_(optimizer)
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.grad is None:
                                continue
                            param.grad.add_(self.c_local[name] - c_global[name])

                if self.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()

                batch_size = x_batch.size(0)
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size
                total_steps += 1

        local_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        result = {
            "state_dict": local_state,
            "loss": total_loss / max(total_samples, 1),
            "num_samples": self.client_data.num_samples,
        }

        if self.fl_method == "scaffold":
            if total_steps == 0:
                raise ValueError("SCAFFOLD requires at least one local optimization step")

            model_state_device = model.state_dict()
            c_local_new = {}

            with torch.no_grad():
                for name in model_state_device.keys():
                    if not torch.is_floating_point(model_state_device[name]):
                        c_local_new[name] = self.c_local[name].detach().clone()
                        continue

                    delta_model = global_state[name] - model_state_device[name]
                    c_local_new[name] = self.c_local[name] - c_global[name] + (delta_model / (total_steps * self.lr))

            c_delta = {}
            for name in c_local_new.keys():
                if torch.is_floating_point(c_local_new[name]):
                    c_delta[name] = (c_local_new[name] - self.c_local[name]).detach().cpu().clone()
                else:
                    c_delta[name] = torch.zeros_like(c_local_new[name]).detach().cpu().clone()

            self.c_local = {k: v.detach().clone() for k, v in c_local_new.items()}
            result["c_delta"] = c_delta

        return result


def build_clients(
    args: argparse.Namespace,
    device: torch.device,
    x_mean: Optional[np.ndarray],
    x_std: Optional[np.ndarray],
    target_mean: Optional[float],
    target_std: Optional[float],
) -> list[BeamClient]:
    train_paths = sorted(glob.glob(os.path.join(args.train_folder, "*.csv")))
    if not train_paths:
        raise FileNotFoundError(f"No train CSV files found in {args.train_folder}")

    clients = []
    for train_csv in train_paths:
        x_train, y_train_label, y_train_db, _, _, _, train_client_ids = load_client_arrays(
            train_csv,
            target_mean=target_mean,
            target_std=target_std,
            assume_powers_are_linear=True,
        )

        if args.standardize_x:
            x_train = normalize_features(x_train, x_mean, x_std)

        client_id = str(train_client_ids[0]) if len(train_client_ids) > 0 else f"client_{len(clients)}"

        train_loader = make_dataloader(
            x_train,
            y_train_db,
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

        clients.append(
            BeamClient(
                client_data=client_data,
                device=device,
                lr=args.lr,
                decay_l2=args.decay_l2,
                local_epochs=args.local_epochs,
                use_amp=args.use_amp,
                fl_method=args.fl_method,
                grad_clip_norm=args.grad_clip_norm,
            )
        )

    return clients


def load_all_test_data(
    test_folder: str,
    standardize_x: bool,
    x_mean: Optional[np.ndarray],
    x_std: Optional[np.ndarray],
    target_mean: Optional[float],
    target_std: Optional[float],
) -> TestDatasetBundle:
    test_paths = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
    if not test_paths:
        raise FileNotFoundError(f"No test CSV files found in {test_folder}")

    xs = []
    ys_label = []
    ys_db = []
    ys_norm_linear = []
    ys_linear = []
    power_sums = []
    client_ids = []

    for test_csv in test_paths:
        x_test, y_test_label, y_test_db, y_test_norm_linear, y_test_linear, ps_test, ids_test = load_client_arrays(
            test_csv,
            target_mean=target_mean,
            target_std=target_std,
            assume_powers_are_linear=True,
        )

        if standardize_x:
            x_test = normalize_features(x_test, x_mean, x_std)

        xs.append(x_test)
        ys_label.append(y_test_label)
        ys_db.append(y_test_db)
        ys_norm_linear.append(y_test_norm_linear)
        ys_linear.append(y_test_linear)
        power_sums.append(ps_test)
        client_ids.extend(ids_test)

    return TestDatasetBundle(
        x_test=np.vstack(xs),
        y_test_label=np.concatenate(ys_label),
        y_test_db=np.vstack(ys_db),
        y_test_norm_linear=np.vstack(ys_norm_linear),
        y_test_linear=np.vstack(ys_linear),
        power_sums=np.concatenate(power_sums),
        client_ids=client_ids,
    )


def update_global_control(
    c_global: dict[str, torch.Tensor],
    client_c_deltas: list[dict[str, torch.Tensor]],
    selected_clients_count: int,
) -> dict[str, torch.Tensor]:
    if not client_c_deltas:
        return c_global

    new_c_global = {}
    for name in c_global.keys():
        if not torch.is_floating_point(c_global[name]):
            new_c_global[name] = c_global[name]
            continue

        avg_delta = sum(delta[name].float() for delta in client_c_deltas) / float(selected_clients_count)
        new_c_global[name] = c_global[name] + avg_delta

    return new_c_global


def federated_train(
    args: argparse.Namespace,
    clients: list[BeamClient],
    global_model: nn.Module,
) -> tuple[nn.Module, Optional[dict[str, torch.Tensor]]]:
    if args.fl_method == "scaffold":
        c_global = init_control_variates_from_model(global_model, device="cpu")
    else:
        c_global = None

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
        client_c_deltas = []

        for client_idx in selected_clients:
            result = clients[client_idx].local_train(
                global_model=global_model,
                global_control=c_global,
                loss_name=args.loss,
            )

            local_state_dicts.append(result["state_dict"])
            local_num_samples.append(result["num_samples"])
            round_losses.append(result["loss"])

            if args.fl_method == "scaffold":
                client_c_deltas.append(result["c_delta"])

        new_global_state = weighted_average_state_dicts(local_state_dicts, local_num_samples)
        global_model.load_state_dict(new_global_state, strict=True)

        if args.fl_method == "scaffold":
            c_global = update_global_control(
                c_global=c_global,
                client_c_deltas=client_c_deltas,
                selected_clients_count=len(selected_clients),
            )

        print(
            f"Round {round_idx:03d}/{args.rounds} | "
            f"method={args.fl_method} | "
            f"clients={len(selected_clients)} | "
            f"loss={weighted_mean(round_losses, local_num_samples):.6f}"
        )

    return global_model, c_global


def evaluate_and_save(
    args: argparse.Namespace,
    global_model: nn.Module,
    test_bundle: TestDatasetBundle,
    device: torch.device,
    x_mean: Optional[np.ndarray],
    x_std: Optional[np.ndarray],
    target_mean: Optional[float],
    target_std: Optional[float],
) -> None:
    global_model.eval()
    global_model.to(device)
    total_inference_time = 0.0
    total_samples = 0

    test_loader = make_dataloader(
        test_bundle.x_test,
        test_bundle.y_test_db,
        test_bundle.y_test_label,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    metrics_rows = []
    sample_offset = 0
    max_top_k = args.max_top_k

    with torch.no_grad():
        for x_batch, _, y_label_batch in test_loader:
            batch_size = x_batch.size(0)
            x_batch = x_batch.to(device, non_blocking=True)
            start_time = time.perf_counter()

            with get_autocast_context(device, args.use_amp and device.type == "cuda"):
                pred = global_model(x_batch)

            if device.type == "cuda":
                torch.cuda.synchronize()  # ensure accurate timing

            end_time = time.perf_counter()

            batch_time = end_time - start_time
            total_inference_time += batch_time
            total_samples += x_batch.size(0)

            pred_np = pred.float().cpu().numpy()
            topk_indices = np.argsort(-pred_np, axis=1)[:, :max_top_k]
            y_true_batch = y_label_batch.numpy()

            for i in range(batch_size):
                global_idx = sample_offset + i
                selected_topk = topk_indices[i].tolist()
                selected_for_power = selected_topk[:args.power_loss_k]
                true_best = int(y_true_batch[i])

                power_loss_db = compute_power_loss_db(
                    selected_for_power,
                    test_bundle.y_test_norm_linear[global_idx],
                    float(test_bundle.power_sums[global_idx]),
                )

                row = {
                    "client_id": test_bundle.client_ids[global_idx],
                    "selected_indices_topk": json.dumps(selected_topk),
                    "true_best": true_best,
                    "predicted_best": int(selected_topk[0]),
                    "best_power": float(test_bundle.y_test_linear[global_idx, true_best]),
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
        "standardize_x": bool(args.standardize_x),
        "standardize_target_db": bool(args.standardize_target_db),
        "power_loss_k": int(args.power_loss_k),
        "optimizer": "adamw",
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
    if target_mean is not None:
        run_config["target_mean_db"] = float(target_mean)
    if target_std is not None:
        run_config["target_std_db"] = float(target_std)

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
    avg_time_per_sample = total_inference_time / max(total_samples, 1)

    print("\nInference timing:")
    print(f"Total inference time: {total_inference_time:.6f} s")
    print(f"Average time per sample: {avg_time_per_sample:.8f} s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated regression of full 64-beam vector in dB scale (FedAvg / SCAFFOLD)"
    )

    parser.add_argument("--train_folder", type=str, default="deepSense/train_sequences_scen1")
    parser.add_argument("--test_folder", type=str, default="deepSense/test_sequences_scen1")
    parser.add_argument("--output_dir", type=str, default="results_regression_db")

    parser.add_argument("--nodes_per_layer", type=int, default=256)
    parser.add_argument("--layers", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--decay_l2", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--fraction_fit", type=float, default=1.0)

    parser.add_argument("--fl_method", type=str, default="scaffold", choices=["fedavg", "scaffold"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1"])

    parser.add_argument("--max_top_k", type=int, default=10)
    parser.add_argument("--power_loss_k", type=int, default=10)

    parser.add_argument("--standardize_x", action="store_true")
    parser.add_argument("--standardize_target_db", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

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
    print("Local optimizer: adamw")

    if args.standardize_x:
        x_mean, x_std = compute_global_feature_stats(args.train_folder)
        print(f"Feature standardization enabled | mean={x_mean.tolist()} | std={x_std.tolist()}")
    else:
        x_mean, x_std = None, None
        print("Feature standardization disabled")

    if args.standardize_target_db:
        target_mean, target_std = compute_global_target_stats(args.train_folder)
        print(f"Target dB standardization enabled | mean={target_mean:.4f} | std={target_std:.4f}")
    else:
        target_mean, target_std = None, None
        print("Target dB standardization disabled")

    global_model = BeamRegressor(
        num_features=2,
        num_outputs=NUM_CLASSES,
        nodes_per_layer=args.nodes_per_layer,
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    clients = build_clients(args, device, x_mean, x_std, target_mean, target_std)
    print(f"Built {len(clients)} training clients")

    test_bundle = load_all_test_data(
        test_folder=args.test_folder,
        standardize_x=args.standardize_x,
        x_mean=x_mean,
        x_std=x_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    print(f"Loaded {len(test_bundle.y_test_label)} test samples from {args.test_folder}")

    global_model, _ = federated_train(args, clients, global_model)

    evaluate_and_save(
        args=args,
        global_model=global_model,
        test_bundle=test_bundle,
        device=device,
        x_mean=x_mean,
        x_std=x_std,
        target_mean=target_mean,
        target_std=target_std,
    )


if __name__ == "__main__":
    main()
