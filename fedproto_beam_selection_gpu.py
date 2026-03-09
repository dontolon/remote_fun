import os
import glob
import json
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Optional

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


class FedProtoNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        nodes_per_layer: int,
        n_layers: int,
        embedding_dim: int,
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
        self.embedding_layer = nn.Linear(in_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.backbone(x)
        embedding = self.embedding_layer(features)
        embedding = F.relu(embedding)
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, x: torch.Tensor):
        embedding = self.encode(x, normalize=True)
        logits = self.classifier(embedding)
        return logits, embedding


@dataclass
class ClientData:
    client_id: str
    train_loader: DataLoader
    eval_train_loader: DataLoader
    num_samples: int


class PrototypeAggregator:
    def __init__(self, num_classes: int, embedding_dim: int):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.global_prototypes: Optional[np.ndarray] = None
        self.global_counts = np.zeros(self.num_classes, dtype=np.int64)

    def aggregate(self, local_sums: list[np.ndarray], local_counts: list[np.ndarray]) -> np.ndarray:
        total_sums = np.zeros((self.num_classes, self.embedding_dim), dtype=np.float64)
        total_counts = np.zeros(self.num_classes, dtype=np.int64)

        for sums, counts in zip(local_sums, local_counts):
            total_sums += sums
            total_counts += counts

        prototypes = np.zeros((self.num_classes, self.embedding_dim), dtype=np.float32)
        valid = total_counts > 0
        prototypes[valid] = (total_sums[valid] / total_counts[valid, None]).astype(np.float32)

        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        prototypes = prototypes / np.clip(norms, 1e-12, None)

        self.global_prototypes = prototypes
        self.global_counts = total_counts
        return prototypes


class BeamClient:
    def __init__(
        self,
        client_data: ClientData,
        device: torch.device,
        lr: float,
        decay_l2: float,
        local_epochs: int,
        lambda_proto: float,
        use_amp: bool,
        compile_model: bool,
    ):
        self.client_data = client_data
        self.device = device
        self.lr = lr
        self.decay_l2 = decay_l2
        self.local_epochs = local_epochs
        self.lambda_proto = lambda_proto
        self.use_amp = use_amp and device.type == "cuda"
        self.compile_model = compile_model

    def _build_local_model(self, global_model: nn.Module) -> nn.Module:
        local_model = copy.deepcopy(global_model).to(self.device)
        if self.compile_model and hasattr(torch, "compile"):
            try:
                local_model = torch.compile(local_model)
            except Exception:
                pass
        return local_model

    def local_train(
        self,
        global_model: nn.Module,
        global_prototypes: Optional[np.ndarray],
        global_counts: Optional[np.ndarray],
    ) -> dict:
        model = self._build_local_model(global_model)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.decay_l2)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        if global_prototypes is not None and global_counts is not None:
            proto_tensor = torch.as_tensor(global_prototypes, dtype=torch.float32, device=self.device)
            proto_mask = torch.as_tensor(global_counts > 0, dtype=torch.bool, device=self.device)
        else:
            proto_tensor = None
            proto_mask = None

        total_loss = 0.0
        total_ce = 0.0
        total_reg = 0.0
        total_samples = 0

        for _ in range(self.local_epochs):
            for x_batch, y_batch in self.client_data.train_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    logits, embedding = model(x_batch)
                    ce_loss = criterion(logits, y_batch)
                    reg_loss = self.prototype_regularization(
                        embedding=embedding,
                        labels=y_batch,
                        global_prototypes=proto_tensor,
                        global_mask=proto_mask,
                    )
                    loss = ce_loss + self.lambda_proto * reg_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = x_batch.size(0)
                total_loss += float(loss.detach().item()) * batch_size
                total_ce += float(ce_loss.detach().item()) * batch_size
                total_reg += float(reg_loss.detach().item()) * batch_size
                total_samples += batch_size

        proto_sums, proto_counts = self.compute_local_prototypes(model)

        return {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "proto_sums": proto_sums,
            "proto_counts": proto_counts,
            "loss": total_loss / max(total_samples, 1),
            "ce_loss": total_ce / max(total_samples, 1),
            "reg_loss": total_reg / max(total_samples, 1),
            "num_samples": self.client_data.num_samples,
        }

    def prototype_regularization(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor,
        global_prototypes: Optional[torch.Tensor],
        global_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if global_prototypes is None or global_mask is None:
            return embedding.new_zeros(())

        valid = global_mask[labels]
        if not torch.any(valid):
            return embedding.new_zeros(())

        selected_embeddings = embedding[valid]
        selected_labels = labels[valid]
        target_prototypes = global_prototypes[selected_labels]

        return F.mse_loss(selected_embeddings, target_prototypes, reduction="mean")

    def compute_local_prototypes(self, model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        embedding_dim = model.classifier.in_features
        proto_sums = torch.zeros(NUM_CLASSES, embedding_dim, device=self.device, dtype=torch.float32)
        proto_counts = torch.zeros(NUM_CLASSES, device=self.device, dtype=torch.long)

        with torch.no_grad():
            for x_batch, y_batch in self.client_data.eval_train_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    embedding = model.encode(x_batch, normalize=True)

                for class_id in y_batch.unique():
                    class_index = int(class_id.item())
                    class_mask = y_batch == class_id
                    proto_sums[class_index] += embedding[class_mask].sum(dim=0)
                    proto_counts[class_index] += int(class_mask.sum().item())

        return (
            proto_sums.cpu().numpy().astype(np.float64),
            proto_counts.cpu().numpy().astype(np.int64),
        )


class TestDatasetBundle:
    def __init__(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_test_norm: np.ndarray,
        power_sums: np.ndarray,
        client_ids: list[str],
    ):
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_norm = y_test_norm
        self.power_sums = power_sums
        self.client_ids = client_ids


def load_client_arrays(csv_path: str):
    df = pd.read_csv(csv_path)
    x = df[["gps_lat", "gps_long"]].values.astype(np.float32)

    missing_cols = [col for col in PWR_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing power columns in {csv_path}: {missing_cols}")

    y_vec = df[PWR_COLS].values.astype(np.float32)

    if "power_sum" in df.columns:
        power_sum = df["power_sum"].values.astype(np.float32)
    else:
        power_sum = np.sum(y_vec, axis=1).astype(np.float32)

    y_norm = y_vec / (power_sum[:, None] + 1e-8)
    labels = np.argmax(y_norm, axis=1).astype(np.int64)

    if "client_id" in df.columns:
        client_ids = df["client_id"].astype(str).tolist()
    else:
        client_ids = [os.path.basename(csv_path)] * len(df)

    return x, labels, y_norm, power_sum, client_ids


def make_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
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


def build_clients(args: argparse.Namespace, device: torch.device) -> list[BeamClient]:
    train_paths = sorted(glob.glob(os.path.join(args.train_folder, "*.csv")))
    if not train_paths:
        raise FileNotFoundError(f"No train CSV files found in {args.train_folder}")

    clients = []
    for train_csv in train_paths:
        x_train, y_train, _, _, train_client_ids = load_client_arrays(train_csv)
        client_id = str(train_client_ids[0]) if len(train_client_ids) > 0 else f"client_{len(clients)}"

        train_loader = make_dataloader(
            x_train,
            y_train,
            args.batch_size,
            True,
            args.num_workers,
            device.type == "cuda",
        )
        eval_train_loader = make_dataloader(
            x_train,
            y_train,
            args.batch_size,
            False,
            args.num_workers,
            device.type == "cuda",
        )

        client_data = ClientData(
            client_id=client_id,
            train_loader=train_loader,
            eval_train_loader=eval_train_loader,
            num_samples=len(y_train),
        )

        client = BeamClient(
            client_data=client_data,
            device=device,
            lr=args.lr,
            decay_l2=args.decay_l2,
            local_epochs=args.local_epochs,
            lambda_proto=args.lambda_proto,
            use_amp=args.use_amp,
            compile_model=args.compile_model,
        )
        clients.append(client)

    return clients


def load_all_test_data(test_folder: str) -> TestDatasetBundle:
    test_paths = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
    if not test_paths:
        raise FileNotFoundError(f"No test CSV files found in {test_folder}")

    xs, ys, y_norms, power_sums, client_ids = [], [], [], [], []
    for test_csv in test_paths:
        x_test, y_test, y_test_norm, ps_test, ids_test = load_client_arrays(test_csv)
        xs.append(x_test)
        ys.append(y_test)
        y_norms.append(y_test_norm)
        power_sums.append(ps_test)
        client_ids.extend(ids_test)

    return TestDatasetBundle(
        x_test=np.vstack(xs),
        y_test=np.concatenate(ys),
        y_test_norm=np.vstack(y_norms),
        power_sums=np.concatenate(power_sums),
        client_ids=client_ids,
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
        raise ValueError("No local state dicts provided for FedAvg aggregation")

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


def maybe_get_lambda_proto(args: argparse.Namespace, round_idx: int) -> float:
    if not args.lambda_warmup:
        return args.lambda_proto

    warmup_rounds = max(1, args.lambda_warmup_rounds)
    progress = min(round_idx / warmup_rounds, 1.0)
    return float(args.lambda_proto * progress)


def federated_train(
    args: argparse.Namespace,
    clients: list[BeamClient],
    global_model: nn.Module,
) -> tuple[nn.Module, PrototypeAggregator]:
    aggregator = PrototypeAggregator(num_classes=NUM_CLASSES, embedding_dim=args.embedding_dim)

    for round_idx in range(1, args.rounds + 1):
        selected_clients = select_participating_clients(
            num_clients=len(clients),
            fraction_fit=args.fraction_fit,
            seed=args.seed,
            round_idx=round_idx,
        )

        current_lambda_proto = maybe_get_lambda_proto(args, round_idx)

        local_state_dicts = []
        local_num_samples = []
        local_sums = []
        local_counts = []
        round_losses = []
        round_ce_losses = []
        round_reg_losses = []

        for client_idx in selected_clients:
            clients[client_idx].lambda_proto = current_lambda_proto

            result = clients[client_idx].local_train(
                global_model=global_model,
                global_prototypes=aggregator.global_prototypes,
                global_counts=aggregator.global_counts,
            )

            local_state_dicts.append(result["state_dict"])
            local_num_samples.append(result["num_samples"])
            local_sums.append(result["proto_sums"])
            local_counts.append(result["proto_counts"])
            round_losses.append(result["loss"])
            round_ce_losses.append(result["ce_loss"])
            round_reg_losses.append(result["reg_loss"])

        new_global_state = weighted_average_state_dicts(local_state_dicts, local_num_samples)
        global_model.load_state_dict(new_global_state, strict=True)

        aggregator.aggregate(local_sums, local_counts)
        valid_classes = int((aggregator.global_counts > 0).sum())

        print(
            f"Round {round_idx:03d}/{args.rounds} | "
            f"clients={len(selected_clients)} | "
            f"lambda_proto={current_lambda_proto:.4f} | "
            f"loss={np.mean(round_losses):.4f} | "
            f"ce={np.mean(round_ce_losses):.4f} | "
            f"reg={np.mean(round_reg_losses):.4f} | "
            f"valid_classes={valid_classes}"
        )

    return global_model, aggregator


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


def evaluate_and_save(
    args: argparse.Namespace,
    global_model: nn.Module,
    aggregator: PrototypeAggregator,
    test_bundle: TestDatasetBundle,
    device: torch.device,
) -> None:
    if aggregator.global_prototypes is None:
        raise RuntimeError("Global prototypes are not available. Training did not run correctly.")

    global_model.eval()
    global_model.to(device)

    global_prototypes = torch.as_tensor(aggregator.global_prototypes, dtype=torch.float32, device=device)
    valid_mask = torch.as_tensor(aggregator.global_counts > 0, dtype=torch.bool, device=device)

    if args.use_classifier_head:
        print("Evaluation mode: classifier head")
    else:
        print("Evaluation mode: nearest prototype")

    test_loader = make_dataloader(
        test_bundle.x_test,
        test_bundle.y_test,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
    )

    metrics_rows = []
    max_top_k = args.max_top_k
    sample_offset = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            batch_size = x_batch.size(0)
            x_batch = x_batch.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(args.use_amp and device.type == "cuda")):
                if args.use_classifier_head:
                    logits, _ = global_model(x_batch)
                    scores = logits.float()
                    scores[:, ~valid_mask] = -torch.inf
                    topk_indices = torch.topk(scores, k=max_top_k, largest=True, dim=1).indices.cpu().numpy()
                else:
                    embedding = global_model.encode(x_batch, normalize=True)
                    distances = torch.cdist(embedding.float(), global_prototypes.float(), p=2.0).pow(2)
                    distances[:, ~valid_mask] = torch.inf
                    topk_indices = torch.topk(distances, k=max_top_k, largest=False, dim=1).indices.cpu().numpy()

            y_true_batch = y_batch.numpy()

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
                    "best_power": float(test_bundle.y_test_norm[global_idx, true_best] * test_bundle.power_sums[global_idx]),
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
        "num_train_clients": int(len(glob.glob(os.path.join(args.train_folder, '*.csv')))),
        "lambda_proto": float(args.lambda_proto),
        "embedding_dim": int(args.embedding_dim),
        "power_loss_k": int(args.power_loss_k),
        "evaluation_mode": "classifier_head" if args.use_classifier_head else "nearest_prototype",
    }

    for k in range(1, max_top_k + 1):
        summary[f"top_{k}_accuracy"] = float(metrics_df[f"hit@{k}"].mean())

    summary[f"mean_power_loss_db_top{args.power_loss_k}"] = float(
        metrics_df[f"power_loss_db_top{args.power_loss_k}"].mean()
    )

    summary_df = pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()])
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model_path = os.path.join(args.output_dir, "global_model.pt")
    torch.save(global_model.state_dict(), model_path)

    proto_path = os.path.join(args.output_dir, "global_prototypes.npy")
    np.save(proto_path, aggregator.global_prototypes)

    counts_path = os.path.join(args.output_dir, "global_proto_counts.npy")
    np.save(counts_path, aggregator.global_counts)

    print(f"\nPer-sample metrics saved to {metrics_path}")
    print(f"Summary saved to {summary_path}")
    print(f"Run config saved to {config_path}")
    print(f"Global model saved to {model_path}")
    print(f"Global prototypes saved to {proto_path}")
    print(f"Prototype counts saved to {counts_path}")

    print("\nTop-k accuracy summary:")
    for k in range(1, max_top_k + 1):
        print(f"Top-{k} accuracy: {summary[f'top_{k}_accuracy']:.4f}")
    print(f"Mean power loss (top {args.power_loss_k}): {summary[f'mean_power_loss_db_top{args.power_loss_k}']:.4f} dB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedAvg + FedProto beam selection from GPS positions")

    parser.add_argument("--train_folder", type=str, default="deepSense/train_sequences_scen1")
    parser.add_argument("--test_folder", type=str, default="deepSense/test_sequences_scen1")
    parser.add_argument("--output_dir", type=str, default="results_ssh")

    parser.add_argument("--nodes_per_layer", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--decay_l2", type=float, default=1e-5)

    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--fraction_fit", type=float, default=1.0)

    parser.add_argument("--lambda_proto", type=float, default=0.05)
    parser.add_argument("--lambda_warmup", action="store_true")
    parser.add_argument("--lambda_warmup_rounds", type=int, default=20)

    parser.add_argument("--max_top_k", type=int, default=10)
    parser.add_argument("--power_loss_k", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--use_classifier_head", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_top_k < 1:
        raise ValueError("max_top_k must be at least 1")
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

    global_model = FedProtoNet(
        num_features=2,
        num_classes=NUM_CLASSES,
        nodes_per_layer=args.nodes_per_layer,
        n_layers=args.layers,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    if args.compile_model and hasattr(torch, "compile"):
        try:
            global_model = torch.compile(global_model)
        except Exception:
            pass

    clients = build_clients(args, device)
    print(f"Built {len(clients)} training clients")

    test_bundle = load_all_test_data(args.test_folder)
    print(f"Loaded {len(test_bundle.y_test)} test samples from {args.test_folder}")

    global_model, aggregator = federated_train(args, clients, global_model)
    evaluate_and_save(args, global_model, aggregator, test_bundle, device)


if __name__ == "__main__":
    main()