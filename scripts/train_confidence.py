# created by clay
'''
The script to parse args and train the qFitLatent confidence model: predicts
per-residue log1p(W1), the Wasserstein distance between a frozen qFitLatent
checkpoint's predicted chi mixture and the true qFit ensemble. Same
backbone+sequence input as qFitLatent; labels come from
scripts/make_w1_labels.py.
'''

import argparse
from pathlib import Path
import torch

from qfit_latent.data.data import ConfidenceDataset, load_subset_paths, load_w1_table
from qfit_latent.model import ConfidenceModel
from qfit_latent.utils import train_confidence

# run the training based on the parsed args
def run_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Get the index of the currently selected device
    idx = torch.cuda.current_device()

    # get name and hardware properties
    gpu_name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)

    # get the vram (converted to GB)
    vram_gb = props.total_memory / (1024 ** 3)

    # print model specs to console
    print(f"device: {device}")
    print(f"  Name: {gpu_name}")
    print(f"  VRAM: {vram_gb:.2f} GB")

    # the structures directory as a path
    structures_dir = Path(args.structures_dir)

    train_paths = load_subset_paths(Path(args.train_csv), structures_dir)
    val_paths = (
        load_subset_paths(Path(args.val_csv), structures_dir)
        if args.val_csv
        else []
    )

    w1_table = load_w1_table(args.w1_csv)
    print(f"w1 labels: {sum(len(v) for v in w1_table.values())} residues "
          f"from {len(w1_table)} pdbs")

    train_dataset = ConfidenceDataset(train_paths, w1_table, max_len=args.max_len)
    val_dataset = ConfidenceDataset(val_paths, w1_table, max_len=args.max_len) if val_paths else None

    print(
        f"train: {len(train_dataset)} structures"
        + (f"  val: {len(val_dataset)}" if val_dataset else "  (no val set)")
    )

    model = ConfidenceModel(
        d_single=args.d_single,
        d_pair=args.d_pair,
        n_ipa=args.n_ipa,
        n_heads=args.n_heads,
        c=args.c,
        dropout=args.dropout,
    )
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path(args.checkpoint_dir)
    start_epoch = 0
    optimizer_state = None
    sched_state = None
    ckpt_path = ckpt_dir / "latest.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        try:
            model.load_state_dict(state["model"])
        except RuntimeError as e:
            print(f"WARNING: checkpoint architecture mismatch — starting fresh. ({e})")
            state = {}
        start_epoch = state.get("epoch", 0)
        optimizer_state = state.get("optimizer")
        sched_state = state.get("scheduler")
        if start_epoch:
            print(f"resumed from epoch {start_epoch}")

    train_confidence(
        model,
        train_dataset,
        val_dataset,
        lr=args.lr,
        epochs=args.epochs,
        device=device,
        checkpoint_dir=ckpt_dir,
        start_epoch=start_epoch,
        save_every=args.save_every,
        loss_csv=args.loss_csv,
        optimizer_state=optimizer_state,
        sched_state=sched_state,
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="train_confidence.py")
    p.add_argument("--structures_dir",
        default="../qfit_data/structures"
    )
    p.add_argument(
        "--train_csv",
        default="qfit_latent/data/train_subset.csv",
    )
    p.add_argument(
        "--val_csv",
        default="qfit_latent/data/val_subset.csv",
    )
    p.add_argument(
        "--w1_csv",
        default="qfit_latent/data/w1_labels.csv",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_len", type=int, default=300)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_single", type=int, default=128)
    p.add_argument("--d_pair", type=int, default=32)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--c", type=int, default=8)
    p.add_argument("--n_ipa", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--checkpoint_dir", default="checkpoints/confidence")
    p.add_argument("--loss_csv", default="logs/confidence.csv")

    args = p.parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
