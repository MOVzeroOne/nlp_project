import pandas as pd 
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm 


path = "./dataset/splits/" 
files = os.listdir(path)

for file_name in tqdm(files,ascii=True):
    path_file = path + file_name 
    df = pd.read_csv(path_file)
    train, test = train_test_split(df, test_size=0.1)

    train.to_csv("./dataset/processed_splits/train_" + file_name,index=False)
    test.to_csv("./dataset/processed_splits/test_" + file_name,index=False)

