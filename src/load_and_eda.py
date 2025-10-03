import pandas as pd

dataset_path = 'data/spam.csv'
df= pd.read_csv(dataset_path, encoding = 'latin-1')

#this will drop extra unnamed columns if they do exist
# v1 = Label( ham or spam)
#v2 = Message
df=df[['v1','v2']]
df.columns=['Label', 'Message']

#and I wrote these just to see the data frame
print(df.head())
print(df.info())
print(df.isna().sum())