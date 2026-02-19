SHEMO_PATH = "shemo"
FEATURE_EXTRACTION_MODEL = "whisper_small"
FEATURES_SAVE_PATH = f"features"
DEVICE = "cuda"

MODEL_SR = 16000

from transformers import WhisperProcessor
import os
import librosa
from tqdm import tqdm
import pandas as pd
from utils.models import WhisperFeatureExtractor
import torch

def extract_features(
        DATA_SET,
        FEATURE_EXTRACTION_MODEL,
        REPRESENTATION_LAYER
):
    
    FEATURES_SAVE_PATH = f"features/{DATA_SET}"
    
    model  = WhisperFeatureExtractor(pretrained_model=FEATURE_EXTRACTION_MODEL, feature_layer=REPRESENTATION_LAYER).to(DEVICE)
    processor = WhisperProcessor.from_pretrained(FEATURE_EXTRACTION_MODEL)

    ############SHEMO
    if not os.path.exists(FEATURES_SAVE_PATH):
        os.makedirs(FEATURES_SAVE_PATH)
        print("The new directory for saving features is created!")
    for root, folders, files in os.walk(SHEMO_PATH):
        for filename in tqdm(files):
            file_path = os.path.join(root, filename)
            audio_data, _ = librosa.load(file_path, sr=MODEL_SR)
            features = processor(audio_data, sampling_rate=MODEL_SR, return_tensors="pt").input_features
            representations = model(features.to(DEVICE))

            for layer, features in representations.items():
                features  = features.detach().to("cpu")
                torch.save(obj=features, f=f"{FEATURES_SAVE_PATH}/{filename.replace('.wav', '')}_{DATA_SET}_{FEATURE_EXTRACTION_MODEL}_{layer.replace('representations', '')}audio_features.pt")