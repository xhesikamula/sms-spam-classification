import joblib

#I AM ONLY USING THE NAIVE_BAYES JUST BECAUSE IT PERFORMED BETTER
#we are loading hte trained model and vectorizer
nb_model = joblib.load("../models/naive_bayes.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

#this function here to predict if a text is ham or spam
def predict_message(text):
    #this here to transform the input text with the trained TF-IDF
    text_tfidf = vectorizer.transform([text])

    #to make the prediction
    prediction = nb_model.predict(text_tfidf)[0]
    return prediction


#to test it in the terminal here
if __name__ == "__main__":
    while True:
        user_input=input("Enter a message(or type 'quit' to exit):")
        if user_input.lower() == "quit":
            break
        result = predict_message(user_input)
        print(f"Prediction: {result}\n")


