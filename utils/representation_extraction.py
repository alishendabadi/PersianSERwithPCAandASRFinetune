from constants import *
from configs import *
from transformers import WhisperProcessor
import os
import librosa
from tqdm import tqdm
from utils.models import WhisperFeatureExtractor
import torch

def extract_features(
        DATA_SET,
        CHECKPOINTS,
        REPRESENTATION_LAYER,
        FEATURES_SAVE_PATH
):
    
    model  = WhisperFeatureExtractor(pretrained_model=CHECKPOINTS, feature_layer=REPRESENTATION_LAYER).to(DEVICE).float()
    processor = WhisperProcessor.from_pretrained(CHECKPOINTS)

    ############SHEMO
    if not os.path.exists(FEATURES_SAVE_PATH):
        os.makedirs(FEATURES_SAVE_PATH)
        print("The new directory for saving features is created!")
    for root, folders, files in os.walk(SHEMO_PATH):
        for filename in tqdm(files):
            file_path = os.path.join(root, filename)
            audio_data, _ = librosa.load(file_path, sr=MODEL_SR)
            features = processor(audio_data, sampling_rate=MODEL_SR, return_tensors="pt").input_features
            representations = model(features.to(DEVICE).float())

            for layer, features in representations.items():
                features  = features.detach().to("cpu")
                torch.save(obj=features, f=f"{FEATURES_SAVE_PATH}/{filename.replace('.wav', '')}.pt")
