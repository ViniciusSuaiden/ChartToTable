import numpy as np
import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from glob import glob
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, AutoTokenizer
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from sklearn.metrics import r2_score
from torch.nn.utils.rnn import pad_sequence
import math

model_path = '/kaggle/input/google-deplot/google-deplot'
processor = Pix2StructProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
BATCHSIZE = 1
UNFREEZE_START = 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BENETECH_LABELS_FOLDER = "/kaggle/input/benetech-making-graphs-accessible/train/annotations"
B_LABELS_F_LEN = len(os.listdir(BENETECH_LABELS_FOLDER))
BENETECH_IMAGES_FOLDER = "/kaggle/input/benetech-making-graphs-accessible/train/images"
B_IMAGES_F_LEN = len(os.listdir(BENETECH_IMAGES_FOLDER))

def rounded(val):
    try:
        if isinstance(val, (int, float)):
            if val == 0:
                return 0
            else:
                num_int_digits = math.floor(math.log10(abs(val))) + 1
                if num_int_digits >= 5:
                    return round(val)
                else:
                    return round(val, -int(math.floor(math.log10(abs(val))) - (5 - 1)))
        else:
            return val
    except:
        return 0

def axis_result(axis, data):
    values = [rounded(item[axis]) for item in data['data-series']]
    axis_result = f"{axis},"
    for item in values:
        axis_result += f"{item};"   
    axis_result = f"{axis_result[:-1]},{data['chart-type']}"
    return axis_result

def load_labels_from_benetech(low_end=0, high_end=B_LABELS_F_LEN):
    labels = []
    for filename in sorted(os.listdir(BENETECH_LABELS_FOLDER))[low_end:high_end]:
        with open(os.path.join(BENETECH_LABELS_FOLDER,filename)) as f:
            data = json.load(f)
            x_result = axis_result("x", data)
            y_result = axis_result("y", data)
            result = f"{x_result} <0x0A> {y_result}"
            labels.append(result)
    return labels

def load_images_from_benetech(low_end=0, high_end=B_IMAGES_F_LEN):
    images = []
    for filename in sorted(os.listdir(BENETECH_IMAGES_FOLDER))[low_end:high_end]:
        img = os.path.join(BENETECH_IMAGES_FOLDER,filename)
        images.append(img)
    return images

def convert_string(string, chart_type):
    x, y = [], []
    for row in string.split(' <0x0A> '):
        cols = row.split(' | ')
        x.append(cols[0])
        y.append(cols[1])
    output_string = 'x,'
    for el in x:
        output_string += f'{el};'
    output_string = output_string[:-1] + f',{chart_type} <0x0A> y,'
    for el in y:
        output_string += f'{el};'
    output_string = output_string[:-1] + f',{chart_type}'
    return output_string

def get_extra_images():
    folder = "/kaggle/input/benetech-extra-generated-data"
    files = []
    paths = []
    for subfolder in ['graphs_d', 'graphs_h', 'graphs_l', 'graphs_s', 'graphs_v']:
        subfolder_path = os.path.join(folder, subfolder)
        for filename in sorted(os.listdir(subfolder_path))[:8000]:
            file = os.path.join(subfolder, filename)
            files.append(file)
            path = os.path.join(folder, subfolder, filename)
            paths.append(path)
    return files, paths

def get_extra_labels(files):
    df = pd.read_csv('/kaggle/input/benetech-extra-generated-data/metadata.csv')
    df.set_index('file_name', inplace=True)
    converted_strings = df.loc[files].apply(lambda row: convert_string(row.text, row.chart_type), axis=1)
    return list(converted_strings)

def mix_datasets(dataset1, dataset2):
    mixed_dataset = []
    i, j = 0, 0
    while i < len(dataset1) or j < len(dataset2):
        if i < len(dataset1):
            mixed_dataset.append(dataset1[i])
            i += 1
        if j < len(dataset2):
            mixed_dataset.append(dataset2[j])
            mixed_dataset.append(dataset2[j+1])
            j += 2
    return mixed_dataset

def get_data():
    labels = load_labels_from_benetech(B_LABELS_F_LEN//3, 2*(B_LABELS_F_LEN//3))
    images = load_images_from_benetech(B_LABELS_F_LEN//3, 2*(B_LABELS_F_LEN//3))
    return images, labels


def load_pretrained_model():
    model = Pix2StructForConditionalGeneration.from_pretrained('/kaggle/input/deplot-finetuned-7/google-deplot-finetuned')

    trainable_model_weights = False
    for pn, p in model.named_parameters():
        if f'decoder.layer.{UNFREEZE_START}' in pn:
            """start unfreezing layer , the weights are trainable"""
            trainable_model_weights = True
        p.requires_grad = trainable_model_weights
        if p.requires_grad:
            print(f"{pn} is set to be trainable.")

    return model.to(device)

class IMGDataset:
    def __init__(self, image_paths, targets, tokenizer=tokenizer):
        self.images = image_paths
        self.labels = targets
        self.target_tokenizer = tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        label = self.target_tokenizer(self.labels[item], padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=200).input_ids
        return image, label
    
def collate_fn(batch):
    images, labels = zip(*batch)
    inputs = processor(images=list(images), text="Generate underlying data table of the figure below:", return_tensors="pt", font_path="/kaggle/input/arial-font/arial.ttf")
    labels = pad_sequence([torch.squeeze(l) for l in labels], batch_first=True)
    return inputs, labels
    
"""main training"""
image_paths, labels = get_data()

nn_model = load_pretrained_model()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, nn_model.parameters()), lr=1e-4, fused=True)
optimizer.zero_grad()
train_dataloader = DataLoader(dataset=IMGDataset(image_paths, labels),
                             batch_size=BATCHSIZE, collate_fn=collate_fn, shuffle=False, num_workers=2)

torch.cuda.empty_cache()
total_loss = 0
for s, batch_data in tqdm(enumerate(train_dataloader)):
    batch_images, batch_targets = batch_data
    batch_images = {k: v.to(device) for k, v in batch_images.items()}
    batch_targets = batch_targets.to(device)
    output = nn_model(**batch_images, labels=batch_targets)
    loss = output.loss
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


    if (s + 1) % 1000 == 0:
        checkpoint = {
            's': s,
            'model_state_dict': nn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }
        torch.save(checkpoint, f'checkpoint_deplot7_2.pth')
        print(f'loss: {total_loss}')
        print(f's: {s}')
        total_loss = 0
        torch.cuda.empty_cache()

nn_model.save_pretrained('google-deplot-finetuned')

!zip -r google-deplot-finetuned.zip google-deplot-finetuned