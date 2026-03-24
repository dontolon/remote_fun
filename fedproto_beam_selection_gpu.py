import os
import glob
import json
import copy
import random
import argparse
from dataclasses import dataclass
from contextlib import nullcontext
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


class BeamClassifier(nn.Module):
    """
    Compact fully-connected classifier for 64-class beam prediction.

    Architecture rationale for the non-IID federated setting:
    - 4 layers of 128 nodes: enough capacity to learn the GPS->beam mapping
      without over-fitting on the small, spatially localised per-client
      datasets typical of the DeepSense6G vehicle route setup.
    - BatchNorm after each linear layer: stabilises training across clients
      whose local distributions differ (different routes, different traffic).
    - Moderate dropout (0.3): regularises without collapsing gradients on
      small local batches.
    - No sigmoid/softmax in forward — raw logits are returned so that
      F.cross_entropy handles the numerically stable log-softmax internally.
    """

    def __init__(
        self,
        num_features: int = 2,
        num_classes: int = NUM_CLASSES,
        nodes_per_layer: int = 128,
        n_layers: int = 4,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        layers = []
        in_dim = num_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, nodes_per_layer))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(nodes_per_layer))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = nodes_per_layer

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_classes)

        # Kaiming initialisation — better than default for ReLU networks
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


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
        y_test_pwr: np.ndarray,
        client_ids: list[str],
    ):
        self.x_test = x_test
        self.y_test_label = y_test_label
        self.y_test_pwr = y_test_pwr   # raw pre-normalized power vectors
        self.client_ids = client_ids


def load_client_arrays(csv_path: str):
    """
    Load pre-normalized data directly from CSV without any transformation.
    The power columns are already in their final form — argmax gives the
    optimal beam label, which is all classification needs.
    """
    df = pd.read_csv(csv_path)

    for col in ("gps_lat", "gps_long"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} is missing column '{col}'")

    missing = [c for c in PWR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing power columns in {csv_path}: {missing}")

    x = df[["gps_lat", "gps_long"]].values.astype(np.float32)
    y_pwr = df[PWR_COLS].values.astype(np.float32)
    y_label = np.argmax(y_pwr, axis=1).astype(np.int64)

    client_ids = (
        df["client_id"].astype(str).tolist()
        if "client_id" in df.columns
        else [os.path.basename(csv_path)] * len(df)
    )

    return x, y_label, y_pwr, client_ids



def make_dataloader(
    x: np.ndarray,
    y_label: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y_label))
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
    k = max(1, int(np.ceil(fraction_fit * num_clients)))
    return sorted(rng.sample(range(num_clients), k))


def weighted_average_state_dicts(
    local_state_dicts: list[dict[str, torch.Tensor]],
    local_num_samples: list[int],
) -> dict[str, torch.Tensor]:
    if not local_state_dicts:
        raise ValueError("No local state dicts provided")

    total = float(sum(local_num_samples))
    avg = {}
    for key in local_state_dicts[0]:
        weighted_sum = None
        for sd, n in zip(local_state_dicts, local_num_samples):
            contrib = sd[key].float() * (float(n) / total)
            weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib
        avg[key] = weighted_sum
    return avg


def weighted_mean(values: list[float], weights: list[int]) -> float:
    if not values:
        return float("nan")
    v = np.asarray(values, np.float64)
    w = np.asarray(weights, np.float64)
    return float(np.sum(v * w) / np.sum(w))


def compute_power_loss_db(
    selected_indices: list[int],
    y_pwr: np.ndarray,
) -> float:
    """
    Power loss in dB between the best selected beam and the true optimal beam.
    y_pwr is the pre-normalized power vector read directly from the CSV.
    """
    if not selected_indices:
        return np.nan
    measured = np.max(y_pwr[np.asarray(selected_indices, np.int64)])
    true_max = np.max(y_pwr)
    if measured <= 0 or true_max <= 0:
        return np.nan
    return float(10.0 * np.log10(true_max / measured))


def clone_state_dict(
    sd: dict[str, torch.Tensor],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    return {k: (v.detach().clone().to(device) if device else v.detach().clone()) for k, v in sd.items()}


def fedprox_penalty(
    model: nn.Module,
    global_params: dict[str, torch.Tensor],
    mu: float,
) -> torch.Tensor:
    """
    (mu/2) * sum_i ||w_i - w_global_i||^2  over all trainable parameters.

    Included in the loss *before* backward so the optimiser sees the full
    penalised landscape — more correct than a post-hoc gradient correction.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            penalty = penalty + torch.sum((param - global_params[name]) ** 2)
    return (mu / 2.0) * penalty


class BeamClient:
    def __init__(
        self,
        client_data: ClientData,
        device: torch.device,
        lr: float,
        decay_l2: float,
        local_epochs: int,
        use_amp: bool,
        grad_clip_norm: float,
        fedprox_mu: float,
        label_smoothing: float,
    ):
        self.client_data = client_data
        self.device = device
        self.lr = lr
        self.decay_l2 = decay_l2
        self.local_epochs = local_epochs
        self.use_amp = use_amp and device.type == "cuda"
        self.grad_clip_norm = grad_clip_norm
        self.fedprox_mu = fedprox_mu
        self.label_smoothing = label_smoothing

    def _build_local_model(self, global_model: nn.Module) -> nn.Module:
        return copy.deepcopy(global_model).to(self.device)

    def local_train(self, global_model: nn.Module) -> dict:
        model = self._build_local_model(global_model)
        model.train()

        optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.decay_l2
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Snapshot global weights for the proximal term
        global_state = clone_state_dict(global_model.state_dict(), device=self.device)

        # Cross-entropy with label smoothing — reduces overconfidence on
        # sparse local label distributions
        ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        total_loss = 0.0
        total_samples = 0

        for _ in range(self.local_epochs):
            for x_batch, y_label_batch in self.client_data.train_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_label_batch = y_label_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with get_autocast_context(self.device, self.use_amp):
                    logits = model(x_batch)
                    task_loss = ce_loss_fn(logits, y_label_batch)
                    prox = fedprox_penalty(model, global_state, self.fedprox_mu)
                    loss = task_loss + prox

                scaler.scale(loss).backward()

                if self.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()

                bs = x_batch.size(0)
                total_loss += float(task_loss.detach().item()) * bs
                total_samples += bs

        return {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "loss": total_loss / max(total_samples, 1),
            "num_samples": self.client_data.num_samples,
        }


def build_clients(
    args: argparse.Namespace,
    device: torch.device,
) -> list[BeamClient]:
    train_paths = sorted(glob.glob(os.path.join(args.train_folder, "*.csv")))
    if not train_paths:
        raise FileNotFoundError(f"No CSVs in {args.train_folder}")

    clients = []
    for csv_path in train_paths:
        x, y_label, _, cids = load_client_arrays(csv_path)
        client_id = cids[0] if cids else f"client_{len(clients)}"

        loader = make_dataloader(
            x, y_label,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        clients.append(BeamClient(
            client_data=ClientData(client_id=client_id, train_loader=loader, num_samples=len(y_label)),
            device=device,
            lr=args.lr,
            decay_l2=args.decay_l2,
            local_epochs=args.local_epochs,
            use_amp=args.use_amp,
            grad_clip_norm=args.grad_clip_norm,
            fedprox_mu=args.fedprox_mu,
            label_smoothing=args.label_smoothing,
        ))

    return clients


def load_all_test_data(test_folder: str) -> TestDatasetBundle:
    test_paths = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
    if not test_paths:
        raise FileNotFoundError(f"No CSVs in {test_folder}")

    xs, ys_label, ys_pwr, cids = [], [], [], []

    for p in test_paths:
        x, y_label, y_pwr, ids = load_client_arrays(p)
        xs.append(x)
        ys_label.append(y_label)
        ys_pwr.append(y_pwr)
        cids.extend(ids)

    return TestDatasetBundle(
        x_test=np.vstack(xs),
        y_test_label=np.concatenate(ys_label),
        y_test_pwr=np.vstack(ys_pwr),
        client_ids=cids,
    )


def federated_train(
    args: argparse.Namespace,
    clients: list[BeamClient],
    global_model: nn.Module,
) -> nn.Module:
    for round_idx in range(1, args.rounds + 1):
        selected = select_participating_clients(
            len(clients), args.fraction_fit, args.seed, round_idx
        )

        local_sds, local_ns, round_losses = [], [], []

        for idx in selected:
            result = clients[idx].local_train(global_model)
            local_sds.append(result["state_dict"])
            local_ns.append(result["num_samples"])
            round_losses.append(result["loss"])

        global_model.load_state_dict(
            weighted_average_state_dicts(local_sds, local_ns), strict=True
        )

        print(
            f"Round {round_idx:03d}/{args.rounds} | "
            f"FedProx mu={args.fedprox_mu} | "
            f"clients={len(selected)} | "
            f"CE loss={weighted_mean(round_losses, local_ns):.6f}"
        )

    return global_model


def evaluate_and_save(
    args: argparse.Namespace,
    global_model: nn.Module,
    test_bundle: TestDatasetBundle,
    device: torch.device,
) -> None:
    global_model.eval()
    global_model.to(device)

    test_loader = make_dataloader(
        test_bundle.x_test,
        test_bundle.y_test_label,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    metrics_rows = []
    sample_offset = 0
    total_inf_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_label_batch in test_loader:
            bs = x_batch.size(0)
            x_batch = x_batch.to(device, non_blocking=True)

            t0 = time.perf_counter()
            with get_autocast_context(device, args.use_amp and device.type == "cuda"):
                logits = global_model(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_inf_time += time.perf_counter() - t0
            total_samples += bs

            topk = np.argsort(-logits.float().cpu().numpy(), axis=1)[:, : args.max_top_k]
            y_true = y_label_batch.numpy()

            for i in range(bs):
                gi = sample_offset + i
                selected_topk = topk[i].tolist()
                true_best = int(y_true[i])

                row = {
                    "client_id": test_bundle.client_ids[gi],
                    "selected_indices_topk": json.dumps(selected_topk),
                    "true_best": true_best,
                    "predicted_best": selected_topk[0],
                    "best_power": float(test_bundle.y_test_pwr[gi, true_best]),
                    f"power_loss_db_top{args.power_loss_k}": compute_power_loss_db(
                        selected_topk[: args.power_loss_k],
                        test_bundle.y_test_pwr[gi],
                    ),
                }
                for k in range(1, args.max_top_k + 1):
                    row[f"hit@{k}"] = int(true_best in selected_topk[:k])

                metrics_rows.append(row)
            sample_offset += bs

    metrics_df = pd.DataFrame(metrics_rows)
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)

    summary = {
        "fl_method": "fedprox",
        "fedprox_mu": float(args.fedprox_mu),
        "label_smoothing": float(args.label_smoothing),
        "nodes_per_layer": int(args.nodes_per_layer),
        "layers": int(args.layers),
        "num_samples": int(len(metrics_df)),
        "num_train_clients": int(len(glob.glob(os.path.join(args.train_folder, "*.csv")))),
    }
    for k in range(1, args.max_top_k + 1):
        summary[f"top_{k}_accuracy"] = float(metrics_df[f"hit@{k}"].mean())
    summary[f"mean_power_loss_db_top{args.power_loss_k}"] = float(
        metrics_df[f"power_loss_db_top{args.power_loss_k}"].mean()
    )

    pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()]).to_csv(
        os.path.join(args.output_dir, "summary.csv"), index=False
    )

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.save(global_model.state_dict(), os.path.join(args.output_dir, "global_model.pt"))

    print(f"\nOutputs → {args.output_dir}")
    print("\nTop-k accuracy:")
    for k in range(1, args.max_top_k + 1):
        print(f"  Top-{k:2d}: {summary[f'top_{k}_accuracy']:.4f}")
    print(f"Mean power loss (top {args.power_loss_k}): {summary[f'mean_power_loss_db_top{args.power_loss_k}']:.4f} dB")
    print(f"Inference: {total_inf_time:.4f}s | {total_inf_time/max(total_samples,1):.8f}s per sample")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedProx beam classification — 64-class GPS-to-beam prediction",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Data
    p.add_argument("--train_folder", default="deepSense/train_sequences_scen1")
    p.add_argument("--test_folder",  default="deepSense/test_sequences_scen1")
    p.add_argument("--output_dir",   default="results_classification_fedprox")

    # Model — tuned for classification on small non-IID clients
    p.add_argument("--nodes_per_layer", type=int,   default=128,
                   help="128 nodes balances capacity vs. overfitting on small client datasets")
    p.add_argument("--layers",          type=int,   default=4,
                   help="4 layers is sufficient for the 2D GPS -> 64-class mapping")
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--no_batchnorm",    action="store_true",
                   help="Disable BatchNorm (not recommended for non-IID FL)")

    # Optimisation
    p.add_argument("--batch_size",      type=int,   default=64,
                   help="Smaller batches help when clients have few samples per class")
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--decay_l2",        type=float, default=1e-4)
    p.add_argument("--grad_clip_norm",  type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing for cross-entropy — reduces overconfidence on sparse labels")

    # Federated
    p.add_argument("--rounds",       type=int,   default=150)
    p.add_argument("--local_epochs", type=int,   default=5)
    p.add_argument("--fraction_fit", type=float, default=1.0)
    p.add_argument("--fedprox_mu",   type=float, default=0.01,
                   help="Proximal penalty. Tune in range 0.001–0.1. Start with 0.01.")

    # Evaluation
    p.add_argument("--max_top_k",    type=int, default=10)
    p.add_argument("--power_loss_k", type=int, default=4)

    # Misc
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--num_workers",   type=int, default=2)
    p.add_argument("--use_amp",       action="store_true")
    p.add_argument("--deterministic", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not (1 <= args.max_top_k <= NUM_CLASSES):
        raise ValueError(f"max_top_k must be in [1, {NUM_CLASSES}]")
    if not (1 <= args.power_loss_k <= args.max_top_k):
        raise ValueError("power_loss_k must be in [1, max_top_k]")
    if not (0 < args.fraction_fit <= 1.0):
        raise ValueError("fraction_fit must be in (0, 1]")
    if args.fedprox_mu < 0:
        raise ValueError("fedprox_mu must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch(device, deterministic=args.deterministic)
    set_seed(args.seed)

    print(f"Device          : {device}")
    print(f"FL method       : FedProx  mu={args.fedprox_mu}")
    print(f"Architecture    : {args.layers} layers x {args.nodes_per_layer} nodes  "
          f"dropout={args.dropout}  batchnorm={not args.no_batchnorm}")
    print(f"Label smoothing : {args.label_smoothing}")
    print(f"Batch size      : {args.batch_size}")

    global_model = BeamClassifier(
        num_features=2,
        num_classes=NUM_CLASSES,
        nodes_per_layer=args.nodes_per_layer,
        n_layers=args.layers,
        dropout=args.dropout,
        use_batchnorm=not args.no_batchnorm,
    ).to(device)

    print(f"Model params    : {sum(p.numel() for p in global_model.parameters()):,}")

    clients = build_clients(args, device)
    print(f"Clients         : {len(clients)}")

    test_bundle = load_all_test_data(args.test_folder)
    print(f"Test samples    : {len(test_bundle.y_test_label)}")

    global_model = federated_train(args, clients, global_model)

    evaluate_and_save(args, global_model, test_bundle, device)


if __name__ == "__main__":
    main()