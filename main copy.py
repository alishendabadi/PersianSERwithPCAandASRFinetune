from utils.reset_seed import reset_seed
from utils.training_utils import train
from utils.testing_utils import test, CrossValidationTestTracker
from utils.preprocessing import read_shemo_build_loader
from utils.representation_loading import collect_all_representations
from utils.cvalidation_utils import print_CV_results
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from utils.models import *

from configs import *
from constants import *


print(f"Running on device: {DEVICE}")
print(f"TensorBoard log dir: {TB_LOG_DIR}")

writer = SummaryWriter(log_dir=TB_LOG_DIR)

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
        use_pca=USE_PCA,
        return_attention_weights=False,
        dropout=DROPOUT,
        dtype=FLOATING_POINT
    ).to(device=DEVICE, dtype=FLOATING_POINT)

    if USE_PCA:
        print(f"  Fitting PCA projector on fold {i+1} training data (no data leak)...")
        train_reps_flat = collect_all_representations(fold["TrainLoader"], REPRESENTATION_CONFIG)
        model.projector.fit(train_reps_flat)
        model = model.to(device=DEVICE, dtype=FLOATING_POINT)  # move newly registered PCA buffers to device/dtype
        writer.add_scalar(f"fold_{i+1}/pca_explained_variance",
                          model.projector.explained_variance_sum, i + 1)

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
                            representation_config=REPRESENTATION_CONFIG,
                            writer=writer,
                            fold_idx=i)

    model.load_state_dict(torch.load(CHECKPOINTS_SAVING_PATH + "/" + model_name))
    model.to(dtype=FLOATING_POINT)
    test_results = test(model=model,
                        loss_fn=LOSS_FUNCTION_CLASS(),
                        test_loader=fold["TestLoader"],
                        device=DEVICE,
                        names_of_categories=names_of_categories,
                        representation_config=REPRESENTATION_CONFIG,
                        writer=writer,
                        fold_idx=i)
    tracker.append(test_results)

print_CV_results(label_to_category=label_to_category, writer=writer, **tracker.__dict__)

# Log all hyperparameters alongside final CV metrics for the HPARAMS tab
writer.add_hparams(
    hparam_dict={
        "architecture":    ARCHITECTURE.name,
        "checkpoint":      os.path.basename(CHECKPOINTS),
        "layer":           LAYER,
        "d_model":         MODEL_EMBEDDING_DIM,
        "heads":           HEAD,
        "hidden_per_head": HIDDEN,
        "lr":              LR,
        "dropout":         DROPOUT,
        "use_pca":         USE_PCA,
        "epochs":          EPOCHS,
        "folds":           NUMBER_OF_FOLDS,
        "floating_point":  _fp_tag,
    },
    metric_dict={
        "hparam/cv_mean_w_accuracy": float(np.mean(tracker.w_accuracy)),
        "hparam/cv_mean_u_accuracy": float(np.mean(tracker.u_accuracy)),
        "hparam/cv_mean_f1":         float(np.mean(tracker.f1)),
        "hparam/cv_mean_loss":       float(np.mean(tracker.loss)),
    }
)

writer.close()
print("TensorBoard writer closed. Run:  tensorboard --logdir ../runs")
