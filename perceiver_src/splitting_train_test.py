import pandas as pd 
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm 

"""
does splitting and blancing the data 
"""


#hyper param
test_size = 0.1 #0.1 is 10 % test size 90 % train size
shuffle = True
#program

path = "./dataset/splits/" 
files = os.listdir(path)

for file_name in tqdm(files,ascii=True):
    #reading in 
    path_file = path + file_name 
    df = pd.read_csv(path_file)
    print("\ndone reading")
    #balancing step
    one_df = df[df["overall"] == 1]
    two_df = df[df["overall"] == 2]
    three_df = df[df["overall"] == 3]
    four_df = df[df["overall"] == 4]
    five_df = df[df["overall"] == 5]

    
    count = min((one_df.count()["reviewText"],two_df.count()["reviewText"],three_df.count()["reviewText"],four_df.count()["reviewText"],five_df.count()["reviewText"]))
    print("count: ", count)

    df = pd.concat([one_df.sample(n = count),two_df.sample(n = count),three_df.sample(n = count),four_df.sample(n = count),five_df.sample(n = count)])
    print("done balancing")
    #splitting step
    train, test = train_test_split(df, test_size=test_size,shuffle=shuffle)

    train.to_csv("./dataset/processed_splits/train_" +"count_"+str(train.count()["reviewText"]) +"_" + file_name,index=False)
    test.to_csv("./dataset/processed_splits/test_" +"count_"+str(test.count()["reviewText"]) + "_"+ file_name ,index=False)
    print("done saving")
