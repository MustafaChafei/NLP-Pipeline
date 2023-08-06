#%%
import string
from bs4 import BeautifulSoup
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import gc
import en_core_web_sm

nlp = en_core_web_sm.load()
# pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Load the dataset
df = pd.read_csv('dataset_score.csv')
df = df.drop_duplicates()
print(df.shape)
df.drop('Unnamed: 0', inplace=True, axis=1)
df = df.rename(columns={'Title':'title','Tag':'tag' ,'Body_x':'text', 'Body_y':'answer'})

df['text'] = df['title'] + ' ' + df['text']

def preprocess_text(text):
    desc = BeautifulSoup(text, "html.parser").getText()
    desc = desc.lower()
    desc = desc.replace('\n', ' ')
    return desc.strip()


df.text = df.text.apply(lambda x: preprocess_text(x))
df.answer = df.answer.apply(lambda x: preprocess_text(x))


with open('df_unprocessed.pkl', 'wb') as f:
    pickle.dump(df, f)
#%%
# Define batch size and initialize lists for storing outputs
batch_size = 16  
outputs = []
attention_masks = {'attention_mask': []}

# Process samples in batches with tqdm for process tracking
for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df[i:i+batch_size]

    # corpus embeddings 
    corp_tokens = {'input_ids':[], 'attention_mask': []}
    for text in batch_df.text:
        new_tokens = tokenizer.encode_plus(text, max_length=512, 
                                            truncation=True, 
                                            padding='max_length',
                                            return_tensors='pt')
        corp_tokens['input_ids'].append(new_tokens['input_ids'][0])
        corp_tokens['attention_mask'].append(new_tokens['attention_mask'][0])
        #attention_masks.append(new_tokens['attention_mask'][0])
        #attention_masks['attention_mask'].append(new_tokens['attention_mask'][0])
    corp_tokens['input_ids'] = torch.stack(corp_tokens['input_ids'])
    corp_tokens['attention_mask']= torch.stack(corp_tokens['attention_mask'])
    attention_masks['attention_mask'].append(corp_tokens['attention_mask']) 
    #attention_masks['attention_mask'] =torch.stack(corp_tokens['attention_mask'])
    #attention_masks = torch.stack(attention_masks)
    with torch.no_grad():
        batch_outputs = model(**corp_tokens)

    
    outputs.append(batch_outputs.last_hidden_state)

    del batch_df
    
    del new_tokens
    del batch_outputs

    gc.collect()
    torch.cuda.empty_cache()



#%%    
# Concatenate the outputs
outputs = torch.cat(outputs, dim=0)
attention_masks['attention_mask'] = torch.cat(attention_masks['attention_mask'], dim=0)
attention = attention_masks['attention_mask']
mask = attention.unsqueeze(-1).expand(outputs.shape).float()
mask_embeddings = outputs * mask
summed_embeddings = torch.sum(mask_embeddings,1)
counts = torch.clamp(mask.sum(1), min = 1e-9)
mean_pooled = summed_embeddings / counts
mean_pooled = mean_pooled.detach().numpy()


# Serialize the outputs
with open('corpus_outputs_unprocessed.pkl', 'wb') as f:
    pickle.dump(outputs, f)

with open('attention_mask_unprocessed.pkl', 'wb') as f:
    pickle.dump(attention_masks, f)

with open('mean_pooled_unprocessed.pkl', 'wb') as f:
    pickle.dump(mean_pooled, f)

# %%
