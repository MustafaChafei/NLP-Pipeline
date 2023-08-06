#%%
import pandas as pd
from bs4 import BeautifulSoup

pd.set_option('display.max_columns', None)
#displays whole text in every column
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
# 10% STACKSAMPLE KAGGLE DATABASE 
tags= pd.read_csv('Tags.csv', encoding="ISO-8859-1")
temp = tags['Tag'].value_counts()
#print(type(temp))
temp = pd.DataFrame(temp)
temp = temp.reset_index()
#%%
question= pd.read_csv('Questions.csv', encoding="ISO-8859-1")
sq = question.sort_values('Score', ascending=False)
sq_2000 = sq.head(50000)
#%%
tags = pd.merge(tags, temp, on='Tag')
tags.where(tags['count']>=47008, inplace=True)

tags.dropna(inplace=True)
print(tags.shape)
tags.head(20)
tags = tags.sort_values(by = 'Id')
tags.head()
#n_tags = tags['Tag'].nunique()
#print(n_tags)
n_ids = tags['Id'].nunique()
print(n_ids)
tag_question_merge = pd.merge(tags, sq_2000, on = 'Id')
#%%
answer= pd.read_csv('Answers.csv', encoding="ISO-8859-1")
#%%

max_score_i = answer.groupby('ParentId')['Score'].idxmax()
max_score_answer = answer.loc[max_score_i]
#print(max_score_answer.head())

print(max_score_answer.columns)
print(tag_question_merge.columns)


tag_question_answer_merge = pd.merge(tag_question_merge, max_score_answer, left_on='Id', right_on='ParentId')
print(tag_question_answer_merge.columns)

tag_question_answer_merge= tag_question_answer_merge[['Title', 'Body_x', 'Tag', 'Body_y']]
tag_question_answer_merge.dropna(subset=['Tag'], inplace=True)

#df = tag_question_answer_merge[tag_question_answer_merge['Score_x'] >= 5]
tag_question_answer_merge.drop_duplicates(inplace=True)
#%%
import pprint as p

p.pprint(tag_question_answer_merge.head(10))

#%%
tag_question_answer_merge.to_csv("dataset_score.csv")
#tag_question_merge = pd.merge(question, tags, on = 'Id')
#tag_question_answer_merge = pd.merge(tag_question_merge, answer, left_on='Id', right_on='ParentId')

#print(tag_question_answer_merge.head())

#%%


