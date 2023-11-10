import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = pickle.load(open('vectorizer.pk1', 'rb'))
model = pickle.load(open('model.pk1', 'rb'))

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps = PorterStemmer()

# Define a text preprocessing function
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [word for word in text if word.isalnum()]

    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]

    return " ".join(text)

st.title("Email Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    print("Transformed Text:",transformed_sms)

    vector_input = tfidf.transform([transformed_sms])

    print("Vector Input:",vector_input)


    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
