import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs("models", exist_ok=True)

df=pd.read_csv('data/spam_cleaned.csv')
# print(df.head())

X = df['CleanedMessage'].astype(str) #these are the features
y = df['Label'] #the target


#this is where the splitting of data is done where 20% of the dataset will be used for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

#converting text into numeric features using TF-IDF
vectorizer=TfidfVectorizer()
X_train_tfidf=vectorizer.fit_transform(X_train)
X_test_tfidf=vectorizer.transform(X_test)

#this si for the part : Train a simple classifier (e.g., Logistic Regression, Naive Bayes, or a small Neural Network)
#Naive Bayes
nb_model=MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

#logistic regression
lr_model=LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

#now this part : Evaluate model accuracy (optionally include precision, recall, F1-score)

print("Naive Bayes:")
print("Accuracy: ", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

print("Logistic Regression:")
print("Accuracy: ", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


#this was a bonus : confusion matrix visualization to compare the models
cm_lr= confusion_matrix(y_test, y_pred_lr, labels=['ham','spam'])
cm_nb= confusion_matrix(y_test, y_pred_nb, labels=['ham','spam'])

fig,axes=plt.subplots(1,2,figsize=(12,5))

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['ham','spam'], yticklabels=['ham','spam'], ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix - Logistic Regression')



sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', xticklabels=['ham','spam'], yticklabels=['ham','spam'], ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix - Naive Bayes')

plt.tight_layout()
plt.show()

#we are saving these trained models and vectorizer for later use
joblib.dump(nb_model, "models/naive_bayes.pkl")
joblib.dump(lr_model, "models/logistic_regression.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModels and vectorizer saved to 'models/' folder.")