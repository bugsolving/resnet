import torch
import pandas as pd
from preprocess import get_files, get_dirs
import numpy as np

def load(img_path, label_path):
    label = pd.read_csv(label_path)
    types = ['Freeform', 'Northwind']
    paths, labels = [], []
    for t in types:
        print(f"Processing type: {t}")
        dirs = get_dirs(img_path + '/' + t)
        for d in dirs:
            print(f"Processing directory: {d}")
            files = get_files(d)
            for file in files:
                print(f"Processing file: {file}")
                no = file
                try:
                    l = label[label['file'] == no]['label'].to_numpy()[0]
                    print(f"Found label: {l} for file {file}")
                    paths.append(d + '/' + file)
                    labels.append(l)
                except IndexError as e:
                    print(f"Label for file {file} not found.")
    return paths, torch.from_numpy(np.array(labels)).view((len(labels), 1))

if __name__ == "__main__":
    img_path = "./test"  # Update this path with the correct one.
    label_path = "./label.csv"  # Ensure this path is correct and accessible.
    paths, labels = load(img_path, label_path)
    print("Final paths:")
    print(paths)
    print("Final labels:")
    print(labels)
