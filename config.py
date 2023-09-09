import torch
import os
import glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/"
PATH_DATASET = os.path.join("data", "DATASET")

# Find the positive and negative weight files
pos_files = glob.glob(f"{MODEL_PATH}/*_multipliers.pth")
pos_files = sorted(pos_files, key=lambda x: float(x.split('/')[-1].split('_')[0]))
neg_files = [f for f in pos_files if float(f.split('/')[-1].split('_')[0]) < 0]
pos_files = [f for f in pos_files if float(f.split('/')[-1].split('_')[0]) > 0]

# Get the latest (or the first in your sorted list) positive and negative weight paths
POS_WEIGHTS_PTH = pos_files[-1] if pos_files else None
NEG_WEIGHTS_PTH = neg_files[-1] if neg_files else None

POS_FACTOR = float(POS_WEIGHTS_PTH.split('/')[-1].split('_')[0]) if POS_WEIGHTS_PTH else None
NEG_FACTOR = float(NEG_WEIGHTS_PTH.split('/')[-1].split('_')[0]) if NEG_WEIGHTS_PTH else None