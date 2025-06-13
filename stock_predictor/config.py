from pathlib import Path
import torch

# -- PATHS --------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

PRICE_CSV = DATA_DIR / "^GSPC.csv"
TWEET_CSV = DATA_DIR / "processed_stockemo_masked.csv"

# -- DATA PROCESSING ----------------------------------------------------------
LOOKBACK    = 6
TEST_RATIO  = 0.33

# -- MODEL HYPERPARAMETERS ----------------------------------------------------
HIDDEN      = 50
LAYERS      = 2
DROPOUT     = 0.15

# -- TRAINING -----------------------------------------------------------------
BATCH_SIZE  = 24
EPOCHS      = 300
LR          = 0.0015
PATIENCE    = 14
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RAND        = 42


