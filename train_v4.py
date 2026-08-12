import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.dataset import KLARestorationDataset
from models.v4_noise_detail import V4NoiseAwareRestoration


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATA PATHS
# ============================================================

DEGRADED_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "degraded"
    / "NoisyLR"
)

GT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gt"
    / "GT"
)

TRAIN_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.txt"
)

VAL_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "val.txt"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "v4_noise_detail"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "v4_noise_detail"
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 4

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0

SEED = 42


# ============================================================
# LOSS WEIGHTS
# ============================================================

PIXEL_WEIGHT = 1.0

EDGE_WEIGHT = 0.15

DETAIL_WEIGHT = 0.03


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("KLA IMAGE RESTORATION - V4 NOISE-AWARE TRAINING")
print("=" * 70)

print()

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ============================================================
# CHECK PATHS
# ============================================================

print()
print("=" * 70)
print("CHECKING DATASET PATHS")
print("=" * 70)

print()
print("Degraded directory:")
print(DEGRADED_DIR)

print()
print("Ground truth directory:")
print(GT_DIR)

print()
print("Training split:")
print(TRAIN_SPLIT)

print()
print("Validation split:")
print(VAL_SPLIT)


for path, name in [
    (DEGRADED_DIR, "Degraded directory"),
    (GT_DIR, "Ground truth directory"),
    (TRAIN_SPLIT, "Training split"),
    (VAL_SPLIT, "Validation split"),
]:

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )


print()
print("✓ All dataset paths exist")


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)


train_dataset = KLARestorationDataset(
    degraded_dir=str(DEGRADED_DIR),
    gt_dir=str(GT_DIR),
    split_file=str(TRAIN_SPLIT)
)


val_dataset = KLARestorationDataset(
    degraded_dir=str(DEGRADED_DIR),
    gt_dir=str(GT_DIR),
    split_file=str(VAL_SPLIT)
)


print()
print(
    "Training samples   :",
    len(train_dataset)
)

print(
    "Validation samples :",
    len(val_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print()
print(
    "Training batches   :",
    len(train_loader)
)

print(
    "Validation batches :",
    len(val_loader)
)

print(
    "Batch size         :",
    BATCH_SIZE
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING V4 MODEL")
print("=" * 70)


model = V4NoiseAwareRestoration(
    in_channels=1,
    out_channels=1,
    channels=64,
    num_blocks=6
).to(DEVICE)


# ============================================================
# PARAMETERS
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print()
print(
    "Total parameters    :",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ============================================================
# SOBEL KERNELS
# ============================================================

sobel_x = torch.tensor(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ],
    dtype=torch.float32
).view(1, 1, 3, 3).to(DEVICE)


sobel_y = torch.tensor(
    [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0]
    ],
    dtype=torch.float32
).view(1, 1, 3, 3).to(DEVICE)


# ============================================================
# LAPLACIAN KERNEL
# ============================================================

laplacian_kernel = torch.tensor(
    [
        [0.0, -1.0, 0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0, 0.0]
    ],
    dtype=torch.float32
).view(1, 1, 3, 3).to(DEVICE)


# ============================================================
# EDGE MAP
# ============================================================

def edge_map(x):

    gx = F.conv2d(
        x,
        sobel_x,
        padding=1
    )

    gy = F.conv2d(
        x,
        sobel_y,
        padding=1
    )

    magnitude = torch.sqrt(
        gx * gx +
        gy * gy +
        1e-6
    )

    return magnitude


# ============================================================
# DETAIL MAP
# ============================================================

def detail_map(x):

    return F.conv2d(
        x,
        laplacian_kernel,
        padding=1
    )


# ============================================================
# V4 LOSS
# ============================================================

def v4_loss(
    prediction,
    target
):

    # --------------------------------------------------------
    # Pixel reconstruction
    # --------------------------------------------------------

    pixel_loss = F.l1_loss(
        prediction,
        target
    )


    # --------------------------------------------------------
    # Edge preservation
    # --------------------------------------------------------

    prediction_edges = edge_map(
        prediction
    )

    target_edges = edge_map(
        target
    )

    edge_loss = F.l1_loss(
        prediction_edges,
        target_edges
    )


    # --------------------------------------------------------
    # Fine detail
    # --------------------------------------------------------

    prediction_detail = detail_map(
        prediction
    )

    target_detail = detail_map(
        target
    )

    detail_loss = F.l1_loss(
        prediction_detail,
        target_detail
    )


    # --------------------------------------------------------
    # Combined loss
    # --------------------------------------------------------

    total_loss = (
        PIXEL_WEIGHT * pixel_loss
        +
        EDGE_WEIGHT * edge_loss
        +
        DETAIL_WEIGHT * detail_loss
    )


    return (
        total_loss,
        pixel_loss,
        edge_loss,
        detail_loss
    )


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    prediction,
    target
):

    prediction = torch.clamp(
        prediction,
        0.0,
        1.0
    )

    target = torch.clamp(
        target,
        0.0,
        1.0
    )


    mse = F.mse_loss(
        prediction,
        target
    )


    if mse.item() <= 1e-12:

        return 100.0


    psnr = (
        10.0
        * torch.log10(
            1.0 / mse
        )
    )


    return psnr.item()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-7
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_psnr = -float("inf")

history = []


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("V4 TRAINING CONFIGURATION")
print("=" * 70)

print()

print(
    "Epochs        :",
    EPOCHS
)

print(
    "Batch size    :",
    BATCH_SIZE
)

print(
    "Learning rate :",
    LEARNING_RATE
)

print(
    "Weight decay  :",
    WEIGHT_DECAY
)

print()

print(
    "Pixel weight  :",
    PIXEL_WEIGHT
)

print(
    "Edge weight   :",
    EDGE_WEIGHT
)

print(
    "Detail weight :",
    DETAIL_WEIGHT
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING V4 TRAINING")
print("=" * 70)

print()


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()


    # ========================================================
    # TRAINING MODE
    # ========================================================

    model.train()


    train_total = 0.0
    train_pixel = 0.0
    train_edge = 0.0
    train_detail = 0.0

    train_batches = 0


    # ========================================================
    # TRAIN BATCHES
    # ========================================================

    for degraded, target in train_loader:

        degraded = degraded.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        prediction = model(
            degraded
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        (
            total_loss,
            pixel_loss,
            edge_loss,
            detail_loss
        ) = v4_loss(
            prediction,
            target
        )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        total_loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        train_total += (
            total_loss.item()
        )

        train_pixel += (
            pixel_loss.item()
        )

        train_edge += (
            edge_loss.item()
        )

        train_detail += (
            detail_loss.item()
        )

        train_batches += 1


    # ========================================================
    # AVERAGE TRAIN LOSS
    # ========================================================

    train_total /= train_batches

    train_pixel /= train_batches

    train_edge /= train_batches

    train_detail /= train_batches


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_total = 0.0

    val_batches = 0

    psnr_total = 0.0

    psnr_images = 0


    with torch.no_grad():

        for degraded, target in val_loader:

            degraded = degraded.to(
                DEVICE,
                non_blocking=True
            )

            target = target.to(
                DEVICE,
                non_blocking=True
            )


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            prediction = model(
                degraded
            )


            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            (
                validation_loss,
                _,
                _,
                _
            ) = v4_loss(
                prediction,
                target
            )


            val_total += (
                validation_loss.item()
            )

            val_batches += 1


            # ------------------------------------------------
            # PSNR
            # ------------------------------------------------

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0
            )


            for i in range(
                prediction.shape[0]
            ):

                image_psnr = calculate_psnr(
                    prediction[i:i + 1],
                    target[i:i + 1]
                )

                psnr_total += image_psnr

                psnr_images += 1


    # ========================================================
    # VALIDATION AVERAGES
    # ========================================================

    val_total /= val_batches

    val_psnr = (
        psnr_total
        / psnr_images
    )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    scheduler.step()


    # ========================================================
    # TIME
    # ========================================================

    epoch_time = (
        time.time()
        - epoch_start
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] | "
        f"Train Loss: {train_total:.6f} | "
        f"Pixel: {train_pixel:.6f} | "
        f"Edge: {train_edge:.6f} | "
        f"Detail: {train_detail:.6f} | "
        f"Val Loss: {val_total:.6f} | "
        f"PSNR: {val_psnr:.3f} dB | "
        f"LR: {current_lr:.2e} | "
        f"Time: {epoch_time:.1f}s"
    )


    # ========================================================
    # HISTORY
    # ========================================================

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_total,
            "train_pixel": train_pixel,
            "train_edge": train_edge,
            "train_detail": train_detail,
            "val_loss": val_total,
            "psnr": val_psnr,
            "lr": current_lr
        }
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_psnr > best_psnr:

        best_psnr = val_psnr


        checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_psnr":
                best_psnr,

            "train_loss":
                train_total,

            "train_pixel_loss":
                train_pixel,

            "train_edge_loss":
                train_edge,

            "train_detail_loss":
                train_detail,

            "val_loss":
                val_total,

            "pixel_weight":
                PIXEL_WEIGHT,

            "edge_weight":
                EDGE_WEIGHT,

            "detail_weight":
                DETAIL_WEIGHT
        }


        best_path = (
            CHECKPOINT_DIR
            / "v4_noise_detail_best.pth"
        )


        torch.save(
            checkpoint,
            best_path
        )


        print(
            f"  ✓ New V4 best model saved "
            f"(PSNR: {best_psnr:.3f} dB)"
        )


    # ========================================================
    # SAVE LATEST
    # ========================================================

    latest_checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_psnr":
            best_psnr
    }


    latest_path = (
        CHECKPOINT_DIR
        / "v4_noise_detail_latest.pth"
    )


    torch.save(
        latest_checkpoint,
        latest_path
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = (
    RESULTS_DIR
    / "training_history.txt"
)


with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V4 NOISE-AWARE DETAIL TRAINING HISTORY\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )


    for item in history:

        f.write(
            f"Epoch {item['epoch']:02d} | "
            f"Train Loss: {item['train_loss']:.6f} | "
            f"Pixel: {item['train_pixel']:.6f} | "
            f"Edge: {item['train_edge']:.6f} | "
            f"Detail: {item['train_detail']:.6f} | "
            f"Val Loss: {item['val_loss']:.6f} | "
            f"PSNR: {item['psnr']:.4f} dB | "
            f"LR: {item['lr']:.3e}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("V4 NOISE-AWARE TRAINING COMPLETE")
print("=" * 70)

print()

print(
    f"Best validation PSNR: "
    f"{best_psnr:.3f} dB"
)

print()

print("Best checkpoint:")
print(
    CHECKPOINT_DIR
    / "v4_noise_detail_best.pth"
)

print()

print("Latest checkpoint:")
print(
    CHECKPOINT_DIR
    / "v4_noise_detail_latest.pth"
)

print()

print("Training history:")
print(
    history_path
)

print()

print("V1, V2 and V3 checkpoints were not modified.")

print("=" * 70)