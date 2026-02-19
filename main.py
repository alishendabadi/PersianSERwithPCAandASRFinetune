from utils.reset_seed import reset_seed
from utils.training_utils import train
from utils.testing_utils import test, CrossValidationTestTracker
from utils.preprocessing import read_shemo_build_loader
from utils.cvalidation_utils import print_CV_results
import torch
import transformers
from utils.models import *
from tqdm import tqdm
from utils.representation_extraction import extract_features
import os

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"
print(DEVICE)


EPOCHS = 30
CHECKPOINTS = "whisper_small"
ARCHITECTURE = QKVMultiheadPoolingSER
DATA_SET = "shemo"
LOSS_FUNCTION_CLASS = torch.nn.CrossEntropyLoss
CHECKPOINTS_SAVING_PATH = "trained_models"


OPTIMIZING_CLASS = torch.optim.AdamW
LR = 1e-3

SCHEDULING_CLASS = transformers.get_cosine_schedule_with_warmup

REPRESENTATION_CONFIG = {"model": CHECKPOINTS,
                         "layer": 12,
                         "path": f"./features/{DATA_SET}",
                         "data_set": DATA_SET}
MODEL_EMBEDDING_DIM = 256
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
NUMBER_OF_FOLDS = 10
HEAD = 6
HIDDEN = 4
DROPOUT = 0.3

label_to_category = {0: "Happy",
                     1: "Neutral",
                     2: "Sad",
                     3: "Angery"}
names_of_categories = label_to_category.values()
number_of_classes = len(names_of_categories)
MODEL_NAME = f"{DATA_SET}_{ARCHITECTURE.name}"

folds = read_shemo_build_loader(number_of_folds=NUMBER_OF_FOLDS,
                            train_batch_size=TRAIN_BATCH_SIZE,
                            test_batch_size=TEST_BATCH_SIZE,
                            shemo_audios_folder_path="./shemo")


LAYER = 8
REPRESENTATION_CONFIG = {"model": CHECKPOINTS,
                        "layer": LAYER,
                        "path": f"./features/{DATA_SET}",
                        "data_set": DATA_SET}

print(f"Layer {REPRESENTATION_CONFIG['layer']}")
tracker = CrossValidationTestTracker()
for i, fold in enumerate(folds):
    print(f"Fold {i+1} ...")
    model_name = MODEL_NAME+f'_layer{REPRESENTATION_CONFIG["layer"]}_head{HEAD}_hidd{HIDDEN}_lr{LR}_fold{i+1}.bin'

    reset_seed(42)
    model = ARCHITECTURE(
        whisper_size=CHECKPOINTS,
        number_of_classes=number_of_classes,
        n_heads=HEAD,
        d_model = MODEL_EMBEDDING_DIM,
        hidden_dim_each_head=HIDDEN,
        return_attention_weights=False,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = OPTIMIZING_CLASS(params= model.parameters(), lr=LR)
    scheduler = SCHEDULING_CLASS(optimizer= optimizer,
                    num_training_steps= EPOCHS,
                    num_warmup_steps= EPOCHS // 10)

    training_results = train(model,
                            saving_path = CHECKPOINTS_SAVING_PATH,
                            model_name=model_name,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loader=fold,
                            epochs=EPOCHS,
                            loss_fn=LOSS_FUNCTION_CLASS(),
                            metric_for_save_best="accuracy",
                            device=DEVICE,
                            representation_config=REPRESENTATION_CONFIG)
    
    model.load_state_dict(torch.load(CHECKPOINTS_SAVING_PATH + "/" + model_name))
    test_results = test(model=model,
                        loss_fn=LOSS_FUNCTION_CLASS(),
                        test_loader=fold["TestLoader"],
                        device=DEVICE,
                        names_of_categories=names_of_categories,
                        representation_config=REPRESENTATION_CONFIG)
    tracker.append(test_results)

print_CV_results(label_to_category=label_to_category, **tracker.__dict__)