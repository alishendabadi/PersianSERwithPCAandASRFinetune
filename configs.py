import torch
import transformers
from utils.models import *

EPOCHS = 30
CHECKPOINTS = "whisper_small"
ARCHITECTURE = QKVMultiheadPoolingSER
DATA_SET = "shemo"
LOSS_FUNCTION_CLASS = torch.nn.CrossEntropyLoss


OPTIMIZING_CLASS = torch.optim.AdamW
LR = 1e-3

SCHEDULING_CLASS = transformers.get_cosine_schedule_with_warmup

LAYER = 8
REPRESENTATION_CONFIG = {"model": CHECKPOINTS,
                        "layer": LAYER,
                        "path": f"./features/{DATA_SET}",
                        "data_set": DATA_SET}


MODEL_EMBEDDING_DIM = 256
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
NUMBER_OF_FOLDS = 10
HEAD = 6
HIDDEN = 4
DROPOUT = 0.3