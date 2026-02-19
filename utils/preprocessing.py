
import numpy as np
import pandas as pd
import librosa, os, torch
from torch.utils.data import Dataset
from collections import namedtuple
import re
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.reset_seed import reset_seed

reset_seed(42)

iemofields = ('labels', 'sessions', 'file_names', 'file_paths', 'genders')
IemoInfo = namedtuple('IemoInfo', iemofields, defaults=(None,) * len(iemofields))

def read_iemocap(iemocap_excel_path, iemo_audios_folder_path, model_sr=16000, save_file_names=True, save_file_path=True):


    df = pd.read_excel(iemocap_excel_path, index_col=0)
    df_cleaned = df[df['emotion'].isin(['ang', 'neu', 'hap', 'sad', 'exc'])]

    # whisper_tiny_processor = WhisperProcessor.from_pretrained('whisper_small')
    emo2num = {
        'hap' : 0,
        'neu' : 1,
        'sad' : 2,
        'ang' : 3,
        'exc' : 0
    }

    num2emo = {
        0 : 'Happy',
        1 : 'Neutral',
        2 : 'Sad',
        3 : 'Angery'
    }

    emotions = []
    labels = []
    file_names = []
    file_paths = []
    genders = []
    sessions = []


    for root, folders, files in os.walk(iemo_audios_folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if not filename.startswith('.'):
                title = filename.replace('.wav', '')
                if title in df_cleaned['name'].tolist():
                    file_path = os.path.join(root, filename)

                    emotion = df_cleaned[df_cleaned['name'] == title]['emotion'].item()
                    label = emo2num[emotion]

                    emotions.append(emotion)
                    labels.append(label)
                    if save_file_names == True: file_names.append(filename)
                    else: file_names.append(None)
                    if save_file_path == True: file_paths.append(file_path)
                    else: file_names.append(None)
                    sessions.append(int(title[4]))
                    genders.append(title[-4])

    return IemoInfo(
       labels = labels,
       file_names = file_names,
       file_paths = file_paths,
       sessions = sessions,
       genders = genders
    )



shemofields = ('labels', 'file_names', 'file_paths', 'genders', 'speakers')
ShemoInfo = namedtuple('ShemoInfo', shemofields, defaults=(None,) * len(shemofields))

def read_shemo(shemo_audios_folder_path, model_sr=16000, save_file_names=True, save_file_path=True):

    num2emo = {
        0 : 'Happy',
        1 : 'Neutral',
        2 : 'Sad',
        3 : 'Angery',
        4 : 'Surprized'
    }

    emo2num = {
       'H': 0,
       'N': 1,
       'S': 2,
       'A': 3,
       'W': 4
    }

    emotions = []
    labels = []
    file_names = []
    file_paths = []
    genders = []
    speakers = []

    pattern = r'^...(.)'

    for root, folders, files in os.walk(shemo_audios_folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            match = re.search(pattern, filename)
            if match:
               char = match.group(1)
               if char in ['H', 'N', 'S', 'A']:
                  emotion = char
                  label = emo2num[emotion]
                  emotions.append(emotion)
                  labels.append(label)
                  if save_file_names: file_names.append(filename)
                  else: file_names.append(None)
                  if save_file_path: file_paths.append(file_path)
                  else: file_paths.append(None)
                  genders.append(filename[0])
                  speakers.append(filename[:3])

    return ShemoInfo(
       labels = labels,
       file_names = file_names,
       file_paths = file_paths,
       genders = genders,
       speakers = speakers
    )

meldfields = ('audios', 'labels', 'file_names', 'file_paths')
MeldInfo = namedtuple('MeldInfo', meldfields, defaults=(None,) * len(meldfields))
def read_meld(meld_excel_path, meld_audios_folder_path, model_sr=16000, save_file_names=True, save_file_path=True):


    df = pd.read_csv(meld_excel_path, index_col=0)
    df['filenames'] = "dia" + df['Dialogue_ID'].astype(str) + "_utt" + df['Utterance_ID'].astype(str) + ".mp3"

    num2emo = {
        0 : 'joy',
        1 : 'neutral',
        2 : 'sadness',
        3 : 'anger',
        4 : 'surprise',
        5 : 'disgust',
        6 : 'fear'
    }

    emo2num = {
       'joy': 0,
       'neutral': 1,
       'sadness': 2,
       'anger': 3,
       'surprise': 4,
       'disgust': 5,
       'fear': 6
    }

    audios = []
    emotions = []
    labels = []
    file_names = []
    file_paths = []
    counter= 0


    for root, folders, files in os.walk(meld_audios_folder_path):
        for filename in files:
            if not filename.startswith("."):
                try :
                    file_path = os.path.join(root, filename)
                    audio_data, _ = librosa.load(file_path, sr=model_sr)
                    emotion = df["Emotion"][df['filenames'] == filename].tolist()[0]
                    label = emo2num[emotion]

                    audios.append(audio_data)
                    emotions.append(emotion)
                    labels.append(label)
                    if save_file_names == True: file_names.append(filename)
                    else: file_names.append(None)
                    if save_file_path == True: file_paths.append(file_path)
                    else: file_names.append(None)
                except:
                    counter += 1
    print(f"errors: {counter}")

    return MeldInfo(
       audios= audios,
       labels = labels,
       file_names = file_names,
       file_paths = file_paths
    )

class MyDataset(Dataset):
  def __init__(self, file_names, labels):
    self.file_names = file_names
    self.labels = labels

  def __len__(self):
    return len(self.file_names)

  def __getitem__(self, idx):
    file_name = self.file_names[idx]
    label = torch.tensor(self.labels[idx])

    return {
        'file_names': file_name, #squeezing is done to unbatch data and remove the vatch dimention (which is 1 here)
        'labels': label,
    }
  
def read_shemo_build_loader(number_of_folds, shemo_audios_folder_path, train_batch_size, test_batch_size):
    print("Creating DataSet ...")
    folds = []
    raw_dataset = read_shemo(shemo_audios_folder_path=shemo_audios_folder_path)
    sessions = [
    ['F01', 'F02', 'F03', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06'], #session1 f3 m6
    ['F04', 'F05', 'F06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12'], #session2 f3 m6
    ['F07', 'F08', 'F09', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18'], #session3 f3 m6
    ['F10', 'F11', 'F12', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24'], #session4 f3 m6
    ['F13', 'F14', 'F15', 'M25', 'M26', 'M27', 'M28', 'M29', 'M30'], #session5 f3 m6
    ['F16', 'F17', 'F18', 'M31', 'M32', 'M33', 'M34', 'M35', 'M36'], #session6 f3 m6
    ['F19', 'F20', 'F21', 'M37', 'M38', 'M39', 'M40', 'M41'], #session7 f3 m5
    ['F22', 'F23', 'F24', 'M42', 'M43', 'M44', 'M45', 'M46'], #session8 f3 m5
    ['F25', 'F26', 'F27', 'M47', 'M48', 'M49', 'M50', 'M51'], #session9 f3 m5
    ['F28', 'F29', 'F30', 'F31', 'M52', 'M53', 'M54', 'M55', 'M56'] #session10 f4 m5
    ]

    for idx in tqdm(range(number_of_folds)):
        session = sessions[idx]
        train_truth = [(i not in session) for i in raw_dataset.speakers]
        test_truth = [(i in session) for i in raw_dataset.speakers]
        train_dataset = MyDataset(file_names=pd.Series(raw_dataset.file_names)[train_truth].to_list(),
                                                            labels=pd.Series(raw_dataset.labels)[train_truth].to_list()
                                                            )
            
        test_dataset = MyDataset(file_names=pd.Series(raw_dataset.file_names)[test_truth].to_list(),
                                                    labels=pd.Series(raw_dataset.labels)[test_truth].to_list()
                                                    )

        loaders = {}
        loaders["TrainLoader"] = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        loaders["TestLoader"] = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        folds.append(loaders)
    print("DataSet Folds for ShEmo are Ready!")

    return folds

def read_iemo_build_loader(number_of_folds, iemocap_excel_path, iemo_audios_folder_path, train_batch_size, test_batch_size):
    folds = []
    raw_dataset = read_iemocap(iemocap_excel_path=iemocap_excel_path, iemo_audios_folder_path=iemo_audios_folder_path)
    for i in range(1, number_of_folds+1):
        session = i
        train_dataset = MyDataset(file_names=pd.Series(raw_dataset.file_names)[np.array(raw_dataset.sessions)!=session].to_list(),
                                                            labels=pd.Series(raw_dataset.labels)[np.array(raw_dataset.sessions)!=session].to_list())
            
        test_dataset = MyDataset(file_names=pd.Series(raw_dataset.file_names)[np.array(raw_dataset.sessions)==session].to_list(),
                                                    labels=pd.Series(raw_dataset.labels)[np.array(raw_dataset.sessions)==session].to_list())
        
        loaders = {}
        loaders["TrainLoader"] = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        loaders["TestLoader"] = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        folds.append(loaders)
    print("DataSet Folds for IEMOCAP are Ready!")
    return folds