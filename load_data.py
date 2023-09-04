from logger import logger

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pathlib
import torch
import librosa
import numpy as np

CURRENT_DIRECTORY = pathlib.Path(__file__).parent.resolve()
DATASET_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..", "data")

OBJECT_NUMBERS = [1, 2, 3, 4, 13, 17, 18, 25, 29, 30, 33, 49, 50, 66, 67, 68, 71, 83, 89, 100]

class CustomDataSet(Dataset):
    def __init__(self, audio, tactile, visual, labels):
        self.audio = audio
        self.tactile = tactile
        self.visual = visual
        self.labels = labels

    def __getitem__(self, index):
        aud = self.audio[index]
        tac = self.tactile[index]
        vis = self.visual[index]
        lab = self.labels[index]
        return aud, tac, vis, lab

    def __len__(self):
        count = len(self.labels)
        assert len(self.tactile) == len(self.labels), "Mismatched tactile examples and label lengths."
        assert len(self.visual) == len(self.labels), "Mismatched visual examples and label lengths."
        assert len(self.audio) == len(self.labels), "Mismatched audioS examples and label lengths."
        return count

def fetch_data():

    logger.log("Start loading the train and validation dataset.")

    TARGET_SIZE = (246, 246)

    audio_train = []
    audio_test = []
    tactile_train = []
    tactile_test = []
    visual_train = []
    visual_test = []
    label_train = []
    label_test = []

    # Initialize transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(TARGET_SIZE),
                                    transforms.ToTensor(),
                                    normalize])

    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/audio/train/{object_number}"
        for audio_files in os.listdir(folder_dir):
            # check if the file ends with wav
            if (audio_files.endswith(".wav")):
                # load the audio
                audio, _ = librosa.load(os.path.join(folder_dir, audio_files), sr=None)
                audio = torch.tensor(audio).unsqueeze(0)  # Add a dimension for the channel
                audio_train.append(torch.tensor(audio))
                label_train.append(object_number)

    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/audio/val/{object_number}"
        for audio_files in os.listdir(folder_dir):
            # check if the file ends with wav
            if (audio_files.endswith(".wav")):
                # load the audio
                audio, _ = librosa.load(os.path.join(folder_dir, audio_files), sr=None)
                audio = torch.tensor(audio).unsqueeze(0)  # Add a dimension for the channel
                audio_test.append(torch.tensor(audio))
                label_test.append(object_number)

    
    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/touch/train/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                tactile_train.append(img_tensor)
 

    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/touch/val/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                tactile_test.append(img_tensor)

    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/vision/train/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_train.append(img_tensor)
 
    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/vision/val/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_test.append(img_tensor)

    logger.log("Finished loading the train and validation dataset.")

    return audio_train, audio_test, tactile_train, tactile_test, visual_train, visual_test, label_train, label_test

def fetch_test_data():
    logger.log("Start loading the test dataset.")
    TARGET_SIZE = (246, 246)

    audio_test = []
    tactile_test = []
    visual_test = []
    label_test = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(TARGET_SIZE),
                                    transforms.ToTensor(),
                                    normalize])

    TARGET_LENGTH = 132300 

    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/audio/test/{object_number}"
        for audio_files in os.listdir(folder_dir):
            if audio_files.endswith(".wav"):
                audio, _ = librosa.load(os.path.join(folder_dir, audio_files), sr=None)
                audio_len = len(audio)

                if audio_len > TARGET_LENGTH:
                    audio = audio[:TARGET_LENGTH] # Truncate the excess
                elif audio_len < TARGET_LENGTH:
                    pad_size = TARGET_LENGTH - audio_len
                    audio = np.pad(audio, (0, pad_size), 'constant') # Pad zeros

                audio = torch.tensor(audio).unsqueeze(0)  # Add a dimension for the channel
                audio_test.append(audio)
                label_test.append(object_number)


    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/touch/test/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                tactile_test.append(img_tensor)
    
    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/vision/test/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_test.append(img_tensor)

    logger.log("Finished loading the test dataset.")

    return audio_test, tactile_test, visual_test, label_test


def get_loader(batch_size):
    audio_train, audio_test, tactile_train, tactile_test, visual_train, visual_test, label_train, label_test = fetch_data()

    logger.log(f"Training on {len(OBJECT_NUMBERS)} objects.")
    logger.log(f"There is audio: {len(audio_train)} train examples and {len(audio_test)} validation examples;")
    logger.log(f"There is touch: {len(tactile_train)} train examples and {len(tactile_test)} validation examples;")
    logger.log(f"There is vision: {len(visual_train)} train examples and {len(visual_test)} validation examples;")

    encoder = LabelEncoder()
    label_train = encoder.fit_transform(label_train)
    label_test = encoder.fit_transform(label_test)
    
    audio = {'train': audio_train, 'test': audio_test}
    tactile = {'train': tactile_train, 'test': tactile_test}
    visual = {'train': visual_train, 'test': visual_test}
    labels = {'train': label_train, 'test': label_test}
    
    dataset = {x: CustomDataSet(audio=audio[x], tactile=tactile[x], visual=visual[x], labels=labels[x]) 
               for x in ['train', 'test']}

    shuffle = {'train': True, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, 
                                shuffle=shuffle[x], num_workers=0) 
                  for x in ['train', 'test']}

    return dataloader

def get_test_loader(batch_size):
    audio_test, tactile_test, visual_test, label_test = fetch_test_data()

    logger.log(f"There is audio: {len(audio_test)} test examples;")
    logger.log(f"There is touch: {len(tactile_test)} test examples;")
    logger.log(f"There is vision: {len(visual_test)} test examples;")

    encoder = LabelEncoder()
    label_test = encoder.fit_transform(label_test)

    audio = {'test': audio_test}
    tactile = {'test': tactile_test}
    visual = {'test': visual_test}
    labels = {'test': label_test}

    test_dataset = {x: CustomDataSet(audio=audio[x], tactile=tactile[x], visual=visual[x], labels=labels[x]) 
               for x in ['test']}
    shuffle = {'test': False}
    test_dataloader = {x: DataLoader(test_dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) 
                  for x in ['test']}

    return test_dataloader