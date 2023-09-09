import os
import glob
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/"
PATH_DATASET = os.path.join("data", "DATASET")

# Find the positive and negative weight files
pos_files = glob.glob(os.path.join(MODEL_PATH, "*_multipliers.pth"))

# Use os.path.basename and os.path.splitext to extract the file name and extension
pos_files = sorted(pos_files, key=lambda x: float(os.path.basename(x).split('_')[0]))
neg_files = [f for f in pos_files if float(os.path.basename(f).split('_')[0]) < 0]
pos_files = [f for f in pos_files if float(os.path.basename(f).split('_')[0]) > 0]

# Get the latest (or the first in your sorted list) positive and negative weight paths
POS_WEIGHTS_PTH = pos_files[-1] if pos_files else None
NEG_WEIGHTS_PTH = neg_files[-1] if neg_files else None

POS_FACTOR = float(os.path.basename(POS_WEIGHTS_PTH).split('_')[0]) if POS_WEIGHTS_PTH else None
NEG_FACTOR = float(os.path.basename(NEG_WEIGHTS_PTH).split('_')[0]) if NEG_WEIGHTS_PTH else None
