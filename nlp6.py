#%%
import en_core_web_sm
import json
import pandas as pd
from bs4 import BeautifulSoup

nlp = en_core_web_sm.load()

df = pd.read_csv('dataset_score.csv')
df = df.drop_duplicates()

print(df.shape)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df.drop('Unnamed: 0', inplace=True, axis=1)
df = df.rename(columns= {'Title':'title','Tag':'tag' ,'Body_x':'text', 'Body_y':'answer'})


#%%
# Remove samples with empty descriptions from dataset
if len(df[df['text']=='']) >0:
    df = df[df['text']!='']
#print(df['tag'].head(15))
#print(df['text'].head())
print(df.shape)
#%%

df = pd.DataFrame({'text':df.text, 'tag':df.tag})
#%%
#---PREPROCESSING----
#--STEP1: lowercase text, remove words that contain numbers
# and remove punctuation
import spacy
import re
import string
import numpy as np
#%%
from langdetect import detect
df['lang'] = df['text'].apply(detect)
df = df[df['lang'] == 'en']

print(df.shape)
#%%
df = pd.DataFrame({'text':df.text, 'tag':df.tag})

def clean(desc):
    desc = BeautifulSoup(desc, "html.parser").getText()
    #desc = re.sub('\[[^\]]*\]', '', desc).strip()
    desc = desc.lower()
    desc = desc.translate(str.maketrans('', '', string.punctuation))
    desc= desc.replace('\n',' ')
    #desc = re.sub('\S*\d\S*\s*', '', desc).strip()
    return desc.strip()

df.text = df.text.apply(lambda x: clean(x))


# %%
#--STEP2:LEMMATIZATION--
#standardize and normalize words
#stopwords: removing common useless words

stopwords = nlp.Defaults.stop_words
def lemmatiser(desc):
    doc = nlp(desc)
    text = [token.lemma_ for token in doc if not token.text in set(stopwords)]
    #text = [str(token) for token in doc if not token.is_stop]
    return ' '.join(text)

df['lemmatized'] = df.text.apply(lambda x:lemmatiser(x))

# %%
'''
#--STEP3:POS TAGS--
#Labeling words with Part of Speech ( NN = nouns ) and extracting all nouns ( for topic modelling and NER)
def pos_labeling(desc):
    doc = nlp(desc)
    text = [token.text for token in doc if token.tag_ =='NN']
    return ' '.join(text)

df['removed_POS'] = df.lemmatized.apply(lambda x: pos_labeling(x))
print(df.head())
'''
#%%
#--FEATURE EXTRACTION--
#TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tfidf = TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english')
document_term_matrix = tfidf.fit_transform(df.lemmatized)

#%%
#--TOPIC MODELLING--
#unsupervised TOPIC CREATION
#Non negative Matrix Factorization (NMF)

from sklearn.decomposition import NMF 
from sklearn.cluster import KMeans
#TODO REPLACE K-MEANS
n_topics = 10
nmf_model = NMF(n_components=n_topics, random_state=40)
#km = KMeans(n_clusters=n_topics)
#document topic matrix (correlation between document and topics)
W1 = nmf_model.fit_transform(document_term_matrix)
#W1 = km.fit_transform(document_term_matrix)

#topic term matrix
H1 = nmf_model.components_
#H1 = km.cluster_centers_

num_words = 15

vocab = np.array(tfidf.get_feature_names_out())
top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_words-1:-1]]
topic_words = ([top_words(t) for t in H1])
topics = [' '.join(t) for t in topic_words]

print(vocab)
for t in topics: 
    print(t)

colnames = ["Topic" + str(i) for i in range(nmf_model.n_components)]

docnames = ["Doc" + str(i) for i in range(len(df.lemmatized))]
df_doc_topic = pd.DataFrame(np.round(W1, 2), columns=colnames, index=docnames)
significant_topic = np.argmax(df_doc_topic.values, axis=1)
df_doc_topic['dominant_topic'] = significant_topic

df['Topic'] = significant_topic

df[['text', 'lemmatized', 'tag', 'Topic']]

t = df[['text', 'lemmatized', 'tag', 'Topic']].groupby('tag').head(30)
print(t[['tag','Topic']].sort_values('tag'))

tag_counts = df.groupby('tag')['Topic'].value_counts()
max_topic_per_tag = tag_counts.groupby('tag').idxmax().apply(lambda x: x[1])

for tag, topic in max_topic_per_tag.items():
    print(f"Tag: {tag}, Topic with the most occurrences: {topic}")

#%%
training_data = df[['lemmatized','tag']]


#%%
mapping = {
    0: 'android',
    1: 'c#',
    2: 'c++',
    3: 'html',
    4: 'ios',
    5: 'java',
    6: 'javascript',
    7: 'jquery',
    8: 'php',
    9: 'python',
}

reverse_mapping = {value: key for key, value in mapping.items()}

training_data['tag'] = training_data['tag'].map(reverse_mapping)
training_data.head(20)
# %%

##--SPLIT DATA IN text AND tag
x = training_data.lemmatized
y = training_data.tag

training_data_temp = pd.DataFrame({'text':x, 'tag':y})
tag_counts = training_data_temp.pivot_table(columns=['tag'], aggfunc='size')
print(tag_counts)


#%%
import matplotlib.pyplot as plt

##--Transform the vector to tf-idf
count_vect = CountVectorizer()
x_vect = count_vect.fit_transform(x)

##--split in train and test datasets
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
x_tfidf = tfidf_transformer.fit_transform(x_vect)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_tfidf,y, test_size=0.2, random_state=40)

#%%

from sklearn.metrics import precision_recall_fscore_support, precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
import inspect

def show_classification_report(model, metrics):
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    y_train_predict_proba = model.predict_proba(x_train)
    y_test_predict_proba = model.predict_proba(x_test)

    acc_train = round(accuracy_score(y_train, y_train_pred),2)
    acc_test = round(accuracy_score(y_test, y_test_pred),2)

    precision_train,recall_train,fscore_train,support_train=precision_recall_fscore_support(y_train,y_train_pred,average='weighted')
    precision_test,recall_test,fscore_test,support_test=precision_recall_fscore_support(y_test,y_test_pred,average='weighted')

    metrics.append(acc_train)
    metrics.append(acc_test)
    metrics.append(np.round(precision_train,2))
    metrics.append(np.round(precision_test,2))
    metrics.append(np.round(recall_train,2))
    metrics.append(np.round(recall_test,2))
    metrics.append(np.round(fscore_train,2))
    metrics.append(np.round(fscore_test,2))
    metrics.append(support_train)
    metrics.append(support_test)

    print("Accuracy of Training Data: ", acc_train)
    print("Accuracy of Test Data: ", acc_test)

    clr_train = classification_report(y_train, y_train_pred, output_dict=True)
    clr_test = classification_report(y_test, y_test_pred, output_dict=True)

    print("Classification Report for Training Data:\n", clr_train)
    print("Classification Report for Test Data:\n", clr_test)
    rtr = pd.DataFrame(clr_train).transpose()
    rte = pd.DataFrame(clr_test).transpose()
    classification_report_df = pd.concat([rtr, rte], axis=1)

    # Name of the Modell Variable
    frame = inspect.currentframe()
    variables = frame.f_back.f_locals.items()
    model_name = [name for name, var in variables if var is model][0]

    # Save Dataframe as Latex Table
    filename = f"classification_report_{model_name}.tex"
    classification_report_df.to_latex(filename)

    fig, ax = plt.subplots(figsize=(12, 8))
    #plt.figure(figsize=(12,8))
    cm = confusion_matrix(y_test, y_test_pred)
    cm_plot = ConfusionMatrixDisplay(cm, display_labels= list(mapping.values()))
    #cm_plot = ConfusionMatrixDisplay(cm, display_labels= model.classes_.map(mapping))
    cm_plot.plot(ax=ax)
    cm_plot.ax_.set_title(f"Confusion Matrix - {model_name}")  
    plt.xticks(rotation=80)

    plt.show()




from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
naive_bayes.fit(x_train, y_train)

metrics1=[]
show_classification_report(naive_bayes, metrics1)


#%%
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(random_state=40, solver='liblinear')
logistic_regression.fit(x_train,y_train)

metrics2=[]
show_classification_report(logistic_regression, metrics2)

#%%

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train, y_train)

metrics3=[]
show_classification_report(dt_classifier, metrics3)


#%%

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train,y_train)

metrics4= []
show_classification_report(rf_classifier, metrics4)


# %%

test_ticket = "How can I use spring with hibernate and JPA?"
test = count_vect.transform([test_ticket])
test_tfidf = tfidf_transformer.transform(test)
nb_pred = logistic_regression.predict(test_tfidf)
nb_proba = logistic_regression.predict_proba(test_tfidf)
top_indices = np.argsort(nb_proba[0])[::-1][:5]  # Change 5 to the desired number of top probabilities

# Retrieve the labels and probabilities for the top indices
top_labels = [mapping[i] for i in top_indices]
top_probabilities = nb_proba[0][top_indices]

print(top_labels)
print(top_probabilities)
#%%
'''
import pickle

with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(logistic_regression, f)

with open('count-vectorizer.pkl','wb') as f:
    pickle.dump(count_vect, f)

with open('tfidf-transformer.pkl','wb') as f:
    pickle.dump(tfidf_transformer, f)
'''