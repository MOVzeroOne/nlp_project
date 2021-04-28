from tqdm import tqdm 
import pandas as pd 

data_reader = pd.read_json("./dataset/All_Amazon_Review_5.json",chunksize=100000,lines=True)

#create empty file (or overwrite file with new empty file)
empty = pd.DataFrame({"reviewText":[],"overall":[]})
empty.to_csv("./dataset/cleaned_step_1.csv",index=False)


#process data
for chunk in tqdm(data_reader,ascii=True):
    #clean up
    
    chunk = chunk.filter(["reviewText", "overall"])
    chunk = chunk.dropna()
    chunk = chunk.reset_index(drop=True)
    #save
    chunk.to_csv("./dataset/cleaned_step_1.csv",mode='a', header=False,index=False)
