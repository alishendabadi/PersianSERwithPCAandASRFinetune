from configs import *
import torch
import os
if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"


CHECKPOINTS_SAVING_PATH = "../trained_models"

label_to_category = {0: "Happy",
                     1: "Neutral",
                     2: "Sad",
                     3: "Angery"}
names_of_categories = label_to_category.values()
number_of_classes = len(names_of_categories)
MODEL_NAME = f"{DATA_SET}_{ARCHITECTURE.name}"

MODEL_SR = 16000

FEATURES_SAVE_PATH = f"../features/{os.path.basename(CHECKPOINTS)}"
SHEMO_PATH = "../shemo"

RUN_NAME   = f"{DATA_SET}_{ARCHITECTURE.name}_{os.path.basename(CHECKPOINTS)}_layer{LAYER}_d{MODEL_EMBEDDING_DIM}_head{HEAD}_hidd{HIDDEN}_lr{LR}"
TB_LOG_DIR = f"../runs/{RUN_NAME}"
