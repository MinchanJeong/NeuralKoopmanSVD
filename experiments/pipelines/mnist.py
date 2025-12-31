import logging
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from .base import BasePipeline
from experiments.factory import get_datasets

logger = logging.getLogger(__name__)


class OracleClassifier(L.LightningModule):
    def __init__(self, num_classes=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # Standard CNN for MNIST
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        # Ensure input has channel dimension: (B, 28, 28) -> (B, 1, 28, 28)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("oracle_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class MnistPipeline(BasePipeline):
    num_classes = 5

    def preprocess(self, overwrite=False):
        data_path = Path(self.cfg.data.path)
        label_path = data_path.parent / "ordered_mnist_labels.npy"

        if data_path.exists() and label_path.exists() and not overwrite:
            logger.info("Ordered MNIST data exists. Skipping.")
            return

        logger.info("Processing Ordered MNIST (Digits 0-4)...")
        dataset = MNIST(root=Path(data_path.parent, "mnist"), train=True, download=True)

        # Filter only labels 0, 1, 2, 3, 4
        mask = dataset.targets < self.num_classes
        images = dataset.data[mask].float() / 255.0
        labels = dataset.targets[mask]

        # Interleave 0->1->2->3->4->0
        indices_by_label = [
            torch.where(labels == i)[0] for i in range(self.num_classes)
        ]
        min_count = min(len(idx) for idx in indices_by_label)

        ordered_indices = torch.stack(
            [
                indices_by_label[digit][i]
                for i in range(min_count)
                for digit in range(self.num_classes)
            ]
        )

        # Select and Save
        ordered_images = images[ordered_indices.flatten()]
        ordered_labels = labels[ordered_indices.flatten()]

        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(data_path, ordered_images.numpy())
        np.save(label_path, ordered_labels.numpy())
        logger.info(f"Saved Ordered MNIST: {ordered_images.shape} to {data_path}")

    def analyze(self, run_dir: Path, output_dir: Path):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Data
        logger.info("Loading Datasets...")
        train_ds, val_ds, collate_fn = get_datasets(self.cfg)

        # Load only validation data; Estimator contains training statistics.
        def to_tensor(ds):
            dl = DataLoader(
                ds,
                batch_size=512,
                num_workers=4,
                shuffle=False,
                collate_fn=collate_fn,
            )
            batches_x, batches_y = [], []
            for batch in dl:
                x, y = batch
                batches_x.append(x)
                batches_y.append(y)
            return torch.cat(batches_x), torch.cat(batches_y)

        # Load Eval Data
        val_x_all, _ = to_tensor(val_ds)
        val_x_all = val_x_all.to(device)
        if val_x_all.ndim == 3:
            val_x_all = val_x_all.unsqueeze(1)

        # Load Train Data (for Oracle)
        train_x_all, _ = to_tensor(train_ds)
        train_x_all = train_x_all.to(device)
        if train_x_all.ndim == 3:
            train_x_all = train_x_all.unsqueeze(1)

        # Labels
        label_path = Path(self.cfg.data.path).parent / "ordered_mnist_labels.npy"
        all_labels = torch.from_numpy(np.load(label_path))
        val_labels = all_labels[val_ds.indices].to(device)
        train_labels = all_labels[train_ds.indices].to(device)

        # 2. Oracle Check (Train if needed)
        oracle = self._get_or_train_oracle(run_dir, train_x_all, train_labels, device)
        oracle.eval()
        class_means = torch.stack(
            [
                train_x_all[train_labels == i].mean(dim=0)
                for i in range(self.num_classes)
            ]
        )

        # 3. Load Model + Estimator
        pl_module, estimator = self.load_model_and_estimator(run_dir)

        if pl_module is None:
            logger.error("Model load failed.")
            return
        if estimator is None:
            logger.error("Estimator load failed.")
            return

        # 4. Evaluation Loop
        n_eval = min(2000, val_x_all.shape[0])
        x0 = val_x_all[:n_eval]
        y0 = val_labels[:n_eval]

        # Pre-compute features
        with torch.no_grad():
            f_x0 = pl_module(x0, lagged=False)
            g_x0 = pl_module(x0, lagged=True)

            if pl_module.svals is not None:
                scale = pl_module.svals.sqrt()
                f_x0 = f_x0 * scale
                g_x0 = g_x0 * scale

            f_x0_np = f_x0.cpu().numpy()
            g_x0_np = g_x0.cpu().numpy()

        steps = list(range(-15, 16))
        steps.remove(0)

        # Define configurations to evaluate
        configs = [
            {"name": "CCA+LoRA", "mode": "aligned", "basis": "f"},
            {"name": "EDMD(f)", "mode": "raw", "basis": "f"},
            {"name": "EDMD(g)", "mode": "raw", "basis": "g"},
        ]

        results = {cfg["name"]: {"acc": {}, "rmse": {}} for cfg in configs}

        for cfg in configs:
            name = cfg["name"]
            mode = cfg["mode"]
            basis = cfg["basis"]

            # Skip if stats are missing
            if (
                mode == "aligned"
                and estimator.operators.get("ali", {}).get("fwd") is None
            ):
                logger.warning(f"Skipping {name}: Aligned operators not found.")
                continue
            if (
                mode == "raw"
                and estimator.operators.get("raw", {}).get(basis, {}).get("fwd") is None
            ):
                logger.warning(
                    f"Skipping {name}: Raw operators for basis {basis} not found."
                )
                continue

            for t in tqdm(steps, desc=f"Eval {name}"):
                try:
                    with torch.no_grad():
                        # Predict
                        pred_np = estimator.predict(
                            f_x0_np, g_x0_np, t=t, mode=mode, basis=basis
                        )

                        x_pred = (
                            torch.from_numpy(pred_np)
                            .float()
                            .to(device)
                            .view(-1, 1, 28, 28)
                        )
                        y_target = (y0 + t) % self.num_classes

                        # Metrics: Accuracy
                        acc = (
                            (oracle(x_pred).argmax(1) == y_target).float().mean().item()
                        )

                        # Metrics: RMSE (Distance to Class Mean)
                        tgt = class_means[y_target]
                        mse = ((x_pred.flatten(1) - tgt.flatten(1)) ** 2).sum(1).mean()
                        rmse = torch.sqrt(mse).item()

                        results[name]["acc"][t] = acc
                        results[name]["rmse"][t] = rmse
                except Exception as e:
                    logger.warning(f"Error in {name} at t={t}: {e}")
                    continue

        # Save Metrics
        with open(output_path / "metrics.json", "w") as f:
            serializable = {
                name: {
                    metric: {str(t): v for t, v in vals.items()}
                    for metric, vals in metrics.items()
                }
                for name, metrics in results.items()
            }
            json.dump(serializable, f, indent=4)

        self._plot_results(results, steps, output_path)
        logger.info(f"Analysis completed. Saved to {output_path}")

    def _save_debug_grid(self, real, pred, path):
        pred = torch.clamp(pred, 0, 1)
        combined = torch.cat([real, pred], dim=3)
        save_image(combined, path, nrow=4, padding=2)

    def _get_or_train_oracle(self, run_dir, x, y, device):
        path = run_dir / "oracle.ckpt"
        oracle = OracleClassifier(num_classes=self.num_classes)
        if path.exists():
            logger.info("Loading existing Oracle classifier...")
            oracle.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        else:
            logger.info("Training Oracle classifier...")
            oracle.to(device)
            ds = TensorDataset(x.cpu(), y.cpu())
            dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)
            trainer = L.Trainer(
                max_epochs=100,
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=False,
            )
            trainer.fit(oracle, dl)
            torch.save({"state_dict": oracle.state_dict()}, path)
        return oracle.to(device)

    def _plot_results(self, results, steps, out_path):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy
        ax = axes[0]
        for mode, data in results.items():
            if not data["acc"]:
                continue
            sorted_steps = sorted(data["acc"].keys())
            vals = [data["acc"][t] for t in sorted_steps]
            ax.plot(sorted_steps, vals, "o-", label=f"{mode} (Acc)")
        ax.set_title("Multistep Accuracy")
        ax.set_xlabel("Time Step (t)")
        ax.set_ylim(0.9, 1.01)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # RMSE
        ax = axes[1]
        for mode, data in results.items():
            if not data["rmse"]:
                continue
            sorted_steps = sorted(data["rmse"].keys())
            vals = [data["rmse"][t] for t in sorted_steps]
            ax.plot(sorted_steps, vals, "s--", label=f"{mode} (RMSE)")
        ax.set_title("Multistep RMSE (vs Mean)")
        ax.set_xlabel("Time Step (t)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path / "eval_plot.png")
        plt.close()
