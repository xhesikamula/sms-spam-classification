import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

df=pd.read_csv('data/spam_cleaned.csv')


#To count the numver of messages per category
print("Number of messages per category:")
print(df['Label'].value_counts())

#I wanted to see it visually
# df['Label'].value_counts().plot(kind='bar', color=['blue','red'])
# plt.title("number of messages per categroy")
# plt.xlabel("Category")
# plt.ylabel("Count")
# plt.show()


#To see the most frequent words in the messages 
all_words=' '.join(df['CleanedMessage'].astype(str)).split() #wrote this to combine all messages into a single list of words

#to count the frequency of each word
word_counts= Counter(all_words)

#this to display the 10 most common words
print("\n10 most common words:")
for word, count in word_counts.most_common(10):
    print(f"{word}:{count}")



