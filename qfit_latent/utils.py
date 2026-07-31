# created by clay
'''
utility functions used for training 
'''

import torch
import csv
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from .data.data import N_AA, N_CHI, SYM_TBL, AA_NAMES
from .loss import ChiARLoss, ConfidenceLoss

def run_epoch(
    model,
    loader,
    device,
    loss_function,
    optimizer=None,
    scheduler=None
):
    # set the boolean for whether its training or not
    training = optimizer is not None
    model.train(training)

    # keep track of loss function
    run_loss = 0.0
    # number of proteins
    n = 0

    aa_sum = torch.zeros(N_AA, dtype=torch.float64)
    aa_count = torch.zeros(N_AA, dtype=torch.float64)

    for sample in tqdm(loader):
        if sample is None:
            continue

        R = sample["R"].to(device)
        t = sample["t"].to(device)
        aa = sample["aa_tokens"].to(device)
        chi_angles = sample["chi_angles"].to(device)
        occupancies = sample["occupancies"].to(device)
        chi_mask = sample["chi_mask"].to(device)

        # skip n=1 rotamers
        if (occupancies > 0).sum(dim=-1).max() < 2:
            continue

        if training:
            optimizer.zero_grad()
        sym_tbl = SYM_TBL.to(device)
        sym_mask = sym_tbl[aa]

        with torch.set_grad_enabled(training):
            # teacher-forced autoregressive head: pass chi_angles for conditioning
            pi, mu, kappa = model(aa, R, t, chi_obs=chi_angles)

            out = loss_function(
                pi, mu, kappa,
                chi_angles, occupancies, chi_mask, sym_mask
            )
            loss = out["loss"]

        if not loss.isfinite():
            continue
        if training and not loss.requires_grad:
            continue

        if training:
            # backprop and step
            loss.backward()
            # gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # iterate
        run_loss += loss.item()
        n += 1

        # store losses per residue
        with torch.no_grad():
            residue_log_prob = out["per_residue_loss"].detach().float().cpu()
            aa_cpu = aa.detach().cpu()

            valid_r = torch.isfinite(residue_log_prob)
            if valid_r.any():
                a = aa_cpu[valid_r].clamp(max=N_AA - 1)
                v = residue_log_prob[valid_r].double()

                in_range = aa_cpu[valid_r] < N_AA
                if in_range.any():
                    aa_sum.scatter_add_(0, a[in_range], v[in_range])
                    aa_count.scatter_add_(
                        0, a[in_range], torch.ones_like(v[in_range])
                    )

    # if none passed return no losses
    if n == 0:
        return None
    
    # avg the losses
    avg_loss = run_loss / n
    # accumulate per aa
    per_aa = (aa_sum / aa_count.clamp(min=1).where(aa_count > 0,
              torch.full_like(aa_sum, float("nan")))).tolist()
    
    return {
        "total": avg_loss,
        "per_aa": per_aa,
    }

def train(
    model,
    train_dataset,
    val_dataset,
    *,
    lr              = 3e-4,
    epochs          = 100,
    device          = "cpu",
    checkpoint_dir  = Path("checkpoints"),
    start_epoch     = 0,
    save_every      = 10,
    loss_csv        = Path("logs/loss.csv"),
    optimizer_state = None,
    sched_state     = None
    ):

    # create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create csv for logging
    loss_csv = Path(loss_csv)
    loss_csv.parent.mkdir(parents=True, exist_ok=True)

    # gpu or cpu
    model.to(device)

    # set up the optimizer (load from previous state if passed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # the loss function (autoregressive vM NLL)
    loss_fn = ChiARLoss()

    # data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True,
        collate_fn=collate,
        num_workers=4
    )
    val_loader   = (DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=collate,
        num_workers=4) if val_dataset is not None else None
    )
    has_val = val_loader is not None

    # set up the scheduler: linear warmup → cosine; rebuild fresh from remaining
    # epochs so resume after architecture/length changes is well-defined
    remaining = max(1, epochs - start_epoch) * len(train_loader)
    warmup = min(500, remaining // 20)
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-3, 1.0, warmup)
    cos  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, remaining - warmup, eta_min=lr / 100
    )
    sched = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], [warmup])

    # determine whether loss csv already exists or not and write header if not
    write_header = not loss_csv.exists() or loss_csv.stat().st_size == 0
    csv_fh = open(loss_csv, "a", newline="")
    writer = csv.writer(csv_fh)
    if write_header:
        writer.writerow(_csv_header(has_val))

    # iterate through epochs and train
    for epoch in range(start_epoch, start_epoch + epochs):

        # run an epoch and return the train loss dict
        tr = run_epoch(model, train_loader, device, loss_fn,
                        optimizer=optimizer, scheduler=sched) #scaler=scaler)
        if tr is None:
            continue

        # run a validation forward pass and return loss
        val = None
        if has_val:
            val = run_epoch(model, val_loader, device, loss_fn)

        # print the losses to console
        if val is not None:
            print(
                f"epoch {epoch+1:>4}  "
                f"total {tr['total']:.4f}  val_total {val['total']:.4f}"
            )
        else:
            print(f"epoch {epoch+1:>4}  total {tr['total']:.4f}")

        # log the losses in the csv
        writer.writerow(_row(epoch + 1, tr, val))
        csv_fh.flush()

        # save the stat of the model/optimizer/scheduler to checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt = {"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": sched.state_dict()}
            torch.save(ckpt, checkpoint_dir / f"epoch_{epoch+1:04d}.pt")
            torch.save(ckpt, checkpoint_dir / "latest.pt")

    # close csv 
    csv_fh.close()
    
def collate(batch):
    valid = [x for x in batch if x is not None]
    return valid[0] if valid else None

def run_epoch_confidence(model, loader, device, loss_function, optimizer=None, scheduler=None):
    # one epoch for ConfidenceModel: same shape as run_epoch, but the model
    # takes no chi_obs and the loss is a scalar regression against W1 labels
    training = optimizer is not None
    model.train(training)

    run_loss = 0.0
    n = 0

    for sample in tqdm(loader):
        if sample is None:
            continue

        R = sample["R"].to(device)
        t = sample["t"].to(device)
        aa = sample["aa_tokens"].to(device)
        w1 = sample["w1"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pred_log_w1 = model(aa, R, t)
            out = loss_function(pred_log_w1, w1)
            loss = out["loss"]

        if not loss.isfinite():
            continue
        if training and not loss.requires_grad:
            continue

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        run_loss += loss.item()
        n += 1

    if n == 0:
        return None
    return {"total": run_loss / n}

def train_confidence(
    model,
    train_dataset,
    val_dataset,
    *,
    lr              = 3e-4,
    epochs          = 100,
    device          = "cpu",
    checkpoint_dir  = Path("checkpoints"),
    start_epoch     = 0,
    save_every      = 10,
    loss_csv        = Path("logs/loss.csv"),
    optimizer_state = None,
    sched_state     = None
    ):
    # same training-loop shape as train(), swapped for the confidence model's
    # simpler (no-teacher-forcing) forward pass and regression loss

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loss_csv = Path(loss_csv)
    loss_csv.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    loss_fn = ConfidenceLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate,
        num_workers=4
    )
    val_loader = (DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
        num_workers=4) if val_dataset is not None else None
    )
    has_val = val_loader is not None

    remaining = max(1, epochs - start_epoch) * len(train_loader)
    warmup = min(500, remaining // 20)
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-3, 1.0, warmup)
    cos  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, remaining - warmup, eta_min=lr / 100
    )
    sched = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], [warmup])

    write_header = not loss_csv.exists() or loss_csv.stat().st_size == 0
    csv_fh = open(loss_csv, "a", newline="")
    writer = csv.writer(csv_fh)
    if write_header:
        writer.writerow(["epoch", "total", "val_total"] if has_val else ["epoch", "total"])

    for epoch in range(start_epoch, start_epoch + epochs):
        tr = run_epoch_confidence(model, train_loader, device, loss_fn,
                                  optimizer=optimizer, scheduler=sched)
        if tr is None:
            continue

        val = run_epoch_confidence(model, val_loader, device, loss_fn) if has_val else None

        if val is not None:
            print(f"epoch {epoch+1:>4}  total {tr['total']:.4f}  val_total {val['total']:.4f}")
        else:
            print(f"epoch {epoch+1:>4}  total {tr['total']:.4f}")

        writer.writerow([epoch + 1, tr["total"]] + ([val["total"]] if val is not None else []))
        csv_fh.flush()

        if (epoch + 1) % save_every == 0:
            ckpt = {"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": sched.state_dict()}
            torch.save(ckpt, checkpoint_dir / f"epoch_{epoch+1:04d}.pt")
            torch.save(ckpt, checkpoint_dir / "latest.pt")

    csv_fh.close()


# format the csv header based on the inputs and whether or not validation is
# being used
def _csv_header(has_val):
    cols = ["epoch", "total"]
    if has_val:
        cols += ["val_total"]
    cols += [f"aa_{a}" for a in AA_NAMES]
    if has_val:
        cols += [f"val_aa_{a}" for a in AA_NAMES]
    return cols

# create a row entry for a given epoch from the loss dicts
def _row(epoch, tr, val):
    row = [epoch, tr["total"]]
    if val is not None:
        row += [val["total"]]
    row += list(tr["per_aa"])
    if val is not None:
        row += list(val["per_aa"])
    return row