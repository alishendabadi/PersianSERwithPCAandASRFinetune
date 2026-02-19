import torch
import os
from utils.representation_extraction import extract_features
from constants import *

def batch_representation_loader(input_names, layer, path, model_size, data_set):
    for input_name in input_names:
        try:
            rep = torch.load(f"{path}/{input_name.replace('.wav', '')}.pt")
        except:
                extract_features(
                    DATA_SET=data_set,
                    CHECKPOINTS=model_size,
                    REPRESENTATION_LAYER=layer,
                    FEATURES_SAVE_PATH
                                )
                rep = torch.load(f"{path}/{input_name.replace('.wav', '')}.pt")
        assert len(rep.shape) == 3 and rep.shape[0] == 1 , "TENSOR IS NOT BATCHED OR BATCH SIZE IS NOT ONE!"
        try:
            input = torch.cat((input, rep), dim=0)
        except:
            input = rep

    return input
