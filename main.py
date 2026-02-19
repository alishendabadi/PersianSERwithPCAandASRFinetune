from utils.reset_seed import reset_seed
from utils.training_utils import train
from utils.testing_utils import test, CrossValidationTestTracker
from utils.preprocessing import read_shemo_build_loader
from utils.cvalidation_utils import print_CV_results
import torch
from utils.models import *

from configs import *
from constants import *


print(f"Running on device: {DEVICE}")



folds = read_shemo_build_loader(number_of_folds=NUMBER_OF_FOLDS,
                            train_batch_size=TRAIN_BATCH_SIZE,
                            test_batch_size=TEST_BATCH_SIZE,
                            shemo_audios_folder_path=SHEMO_PATH)




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