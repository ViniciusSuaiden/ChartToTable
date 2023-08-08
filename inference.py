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

processor = Pix2StructProcessor.from_pretrained('/kaggle/input/google-deplot/google-deplot')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

nn_model = Pix2StructForConditionalGeneration.from_pretrained('/kaggle/input/deplot-finetuned-7-2/google-deplot-finetuned').to(device)

def parse_string(s):
    default_object = {'x': '0;0', 'y': '0;0', 'type': 'line'}
    try:
        lines = s.split(" <0x0A> ")
        result = {}
        for line in lines:
            parts = line.split(",")
            key = parts[0]
            data_parts = parts[1].split(";")
            type_ = parts[-1]
            if key == '' or type_ == '':
                return default_object
            if type_ in ["horizontal_bar", "vertical_bar", "line", "dot", "scatter"]:
                if key == 'y':
                    for i in range(len(data_parts)):
                        try:
                            if math.isnan(float(data_parts[i])):
                                data_parts[i] = '0'
                        except:
                            data_parts[i] = '0'
                    while len(data_parts) < len(result['x']):
                        data_parts.append(data_parts[-1])
            else:
                return default_object
            result[key] = data_parts
        result['x'] = ";".join(result['x'])
        result['y'] = ";".join(result['y'])
        result['type'] = type_
        if 'x' in result and 'y' in result:
            return result
        else:
            return default_object
    except Exception:
        return default_object

    
dfs = []
test_path = "/kaggle/input/benetech-making-graphs-accessible/test/images"
for filename in tqdm(sorted(os.listdir(test_path))):
    image = Image.open(os.path.join(test_path,filename))
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt", font_path="/kaggle/input/arial-font/arial.ttf")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    predictions = nn_model.generate(**inputs, max_new_tokens=512)
    pred = processor.decode(predictions[0], skip_special_tokens=True)
    sub_obj = parse_string(pred)
    sub_df = pd.DataFrame({ 'id': [f'{filename[:-4]}_x', f'{filename[:-4]}_y'], 
                        'data_series': [sub_obj['x'], sub_obj['y']],
                        'chart_type': [sub_obj['type'], sub_obj['type']] }).set_index('id')
    dfs.append(sub_df)

df = pd.concat(dfs)
df.to_csv('submission.csv')