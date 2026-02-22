import torch
import transformers
import os
from utils.models import *

EPOCHS = 30
CHECKPOINTS = "../whisper-small"
ARCHITECTURE = QKVMultiheadPoolingSER
DATA_SET = "shemo"
LOSS_FUNCTION_CLASS = torch.nn.CrossEntropyLoss


OPTIMIZING_CLASS = torch.optim.AdamW
LR = 1e-3

SCHEDULING_CLASS = transformers.get_cosine_schedule_with_warmup

LAYER = 12
REPRESENTATION_CONFIG = {"model": CHECKPOINTS,
                        "layer": LAYER,
                        "path": f"../features/{os.path.basename(CHECKPOINTS)}",
                        "data_set": DATA_SET}


MODEL_EMBEDDING_DIM = 512
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
NUMBER_OF_FOLDS = 10
HEAD = 6
HIDDEN = 4
DROPOUT = 0.3
USE_PCA = True

# Floating-point precision for the model and all input tensors.
# torch.float32  — full precision (default, compatible everywhere)
# torch.float16  — half precision (faster on CUDA, lower VRAM; not recommended on CPU)
FLOATING_POINT = torch.float32
