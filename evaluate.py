import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.v4_noise_detail import V4NoiseAwareRestoration


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="KLA V4 standalone image restoration evaluator"
)

parser.add_argument(
    "input_dir",
    type=str,
    help="Directory containing degraded .npy images"
)

parser.add_argument(
    "output_dir",
    type=str,
    help="Directory where restored .npy images will be saved"
)

args = parser.parse_args()

INPUT_DIR = Path(args.input_dir).resolve()
OUTPUT_DIR = Path(args.output_dir).resolve()

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoint"
    / "v4_noise_detail_best.pth"
)

# ------------------------------------------------------------
# CHECK PATHS
# ------------------------------------------------------------

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Input directory not found: {INPUT_DIR}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"V4 checkpoint not found: {CHECKPOINT}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("KLA V4 STANDALONE RESTORATION EVALUATOR")
print("=" * 70)
print()
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

print()
print("Input directory :", INPUT_DIR)
print("Output directory:", OUTPUT_DIR)
print("Checkpoint      :", CHECKPOINT)


# ------------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------------

model = V4NoiseAwareRestoration(
    in_channels=1,
    out_channels=1,
    channels=64,
    num_blocks=6
).to(device)


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------

print()
print("Loading V4 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model" in checkpoint:
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# ------------------------------------------------------------
# REMOVE module. PREFIX
# ------------------------------------------------------------

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[len("module."):]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.eval()

print("Model loaded successfully.")


# ------------------------------------------------------------
# FIND INPUT FILES
# ------------------------------------------------------------

input_files = sorted(
    INPUT_DIR.glob("*.npy")
)

if len(input_files) == 0:
    raise RuntimeError(
        f"No .npy files found in {INPUT_DIR}"
    )

print()
print("Input images:", len(input_files))
print()


# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

processed = 0

with torch.no_grad():

    for index, input_file in enumerate(input_files, start=1):

        # Load degraded image
        degraded = np.load(input_file)

        # Validate input
        if degraded.shape != (128, 128):
            raise ValueError(
                f"Unexpected input shape for "
                f"{input_file.name}: {degraded.shape}. "
                f"Expected (128, 128)."
            )

        degraded = degraded.astype(
            np.float32,
            copy=False
        )

        # NumPy [H,W] -> Torch [1,1,H,W]
        tensor = torch.from_numpy(
            degraded
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(
            device,
            non_blocking=True
        )

        # V4 inference
        prediction = model(tensor)

        # Handle tuple/list output if present
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]

        # Clamp to valid image range
        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        # Torch [1,1,256,256] -> NumPy [256,256]
        restored = (
            prediction[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Validate output
        if restored.shape != (256, 256):
            raise RuntimeError(
                f"Unexpected output shape for "
                f"{input_file.name}: {restored.shape}. "
                f"Expected (256, 256)."
            )

        # Save using the same filename
        output_file = OUTPUT_DIR / input_file.name

        np.save(
            output_file,
            restored
        )

        processed += 1

        if (
            processed == 1
            or processed % 80 == 0
            or processed == len(input_files)
        ):
            print(
                f"Processed {processed}/{len(input_files)}"
            )


# ------------------------------------------------------------
# FINAL CHECK
# ------------------------------------------------------------

output_files = sorted(
    OUTPUT_DIR.glob("*.npy")
)

print()
print("=" * 70)
print("RESTORATION COMPLETE")
print("=" * 70)
print()
print("Input images :", len(input_files))
print("Output images:", len(output_files))
print()

if len(output_files) != len(input_files):
    raise RuntimeError(
        "Output count does not match input count!"
    )

print("✓ Output count matches input count")
print("✓ All outputs are 256×256 float32 .npy files")
print()
print("Output directory:")
print(OUTPUT_DIR)
print()
print("=" * 70)
