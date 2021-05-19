import pandas as pd 
from transformers import AutoTokenizer
from tqdm import tqdm 

"""
splits data up in sentence length (after tokenization)
so that every file has sentences of lengths up to a specified number 
"""


data = pd.read_csv("./dataset/cleaned_step_1.csv",chunksize=100000) 
voc = AutoTokenizer.from_pretrained('bert-base-cased')
split_list = [100,200,400,800,1600] #in increasing order


dataset_paths = []




for num in split_list:
    """
    overwrites old data and setup of paths
    """

    path = "./dataset/splits/cleaned_" + str(num) +  ".csv"
    empty = pd.DataFrame({"reviewText":[],"overall":[],"length":[]})
    empty.to_csv(path,index=False)\

    dataset_paths.append(path)


for chunk in tqdm(data,ascii=True):
    #data processing    
    #text = line["reviewText"]
    #rating = line["overall"]

    
    chunk["reviewText"] = chunk.applymap(str)

    chunk["length"] = [len(tokens) for tokens in voc(chunk["reviewText"].to_list())['input_ids']]
    
    #sentence_length = voc(text)["input_ids"]
    #print(sentence_length)

  
    #splitting of the data
    

    for i in range(len(split_list)):
        
        data = chunk[chunk["length"] <= split_list[i]]
        data = data.reset_index(drop=True)
        #save
        data.to_csv(dataset_paths[i],mode='a', header=False,index=False)

    


