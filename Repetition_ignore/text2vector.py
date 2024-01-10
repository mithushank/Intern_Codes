import numpy as np
from collections import Counter
import pandas as pd
import time
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

start_time = time.time()
# Load the data set
titles = ['label','title','description']
csv_file_path = 'Intern_codes/Datasets/amazon_review_polarity_csv/train_sample.csv'
df = pd.read_csv(csv_file_path, names = titles )
df = df.iloc[:15]
vector_list =[]
y_train = df['label'].tolist()
pool_of_vector =[]
pool_of_samples=[]
# Test the similarity
def Similarity_check(pool_of_vector ,pool_of_samples, new_vector ,thresh = 0.7):  
    flag = 0  
    if len(pool_of_vector) <= 0:
        pool_of_vector.append(new_vector)
        flag = 1  
    else:    
        for vec in pool_of_vector:
            Similarity_score = cosine_similarity([vec, new_vector])
            # print(Similarity_score,' Finished')
            # print(Similarity_score[0][0],Similarity_score[0][1])
            if Similarity_score[0][0] <= thresh or Similarity_score[0][1] <= thresh :
                pool_of_vector.append(new_vector)  
                flag = 1      
                break
    return pool_of_vector,flag

for index, row in df.iterrows():
    vect_list = []
    for column in titles[1:]:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        tokens = tokenizer(row[column], return_tensors='pt', padding=True, truncation=True)
        outputs = model(**tokens)
        vector_for_text = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        
        vect_list.append(vector_for_text)
    vect_list = (vect_list[0]+vect_list[1])/2
    pool_of_vector,flag =Similarity_check(pool_of_vector,pool_of_samples, vect_list)
    if flag:
        pool_of_samples.append(row[titles[1:]])
 
       
print(len(pool_of_samples))
print('Length of pool of vectors: ',len(pool_of_vector))
print('Length of vectors  in original: ', index)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
