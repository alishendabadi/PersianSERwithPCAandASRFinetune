import torch
import os
from tqdm import tqdm
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


def collect_all_representations(loader, representation_config):
  """
  Load every representation in a DataLoader fold and return a 2D tensor
  suitable for fitting PCA.

  Returns: float32 CPU tensor of shape [N_samples * T, D]
           where N_samples = total samples in loader, T = sequence length,
           D = whisper encoder hidden dim.
  """
  all_reps = []
  for batch in tqdm(loader, desc="  Collecting representations for PCA"):
    reps = batch_representation_loader(
      batch['file_names'],
      representation_config["layer"],
      representation_config["path"],
      representation_config["model"],
      representation_config["data_set"]
    )  # [B, T, D]
    all_reps.append(reps.cpu().float())
  all_reps = torch.cat(all_reps, dim=0)          # [N, T, D]
  return all_reps.view(-1, all_reps.shape[-1])    # [N*T, D]
