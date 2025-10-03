import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

#the dataset
dataset_path = 'data/spam.csv'
df= pd.read_csv(dataset_path, encoding = 'latin-1')
# v1 = Label( ham or spam)
#v2 = Message
df=df[['v1','v2']]
df.columns=['Label', 'Message']


#I'm using NLTK so we should download stopwords first
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    if isinstance(text, str):
    # lowercase
        text = text.lower()
        #this to remove any special characters
        text = re.sub(r'[^a-z0-9\s]', '' , text)
        #to replace any sequence of whitespace with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        #to remove stopwords
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    else:
        return ''

#apply the cleaning
df['CleanedMessage'] =df['Message'].apply(clean_text)

#just to inspect
print(df[['Message', 'CleanedMessage']].head(17))

df.to_csv('data/spam_cleaned.csv', index=False)