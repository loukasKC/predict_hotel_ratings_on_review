#import three libraries for dataframe, data manipulation
import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

#import for loading the models
import joblib

#imports for NLP
import spacy
import nltk
from nltk.corpus import stopwords
import string

#imports to prepare the data for the algorithms
from sklearn.feature_extraction.text import TfidfVectorizer

#imports for models
from sklearn.svm import LinearSVC

def cleanData(reviews):

    #dictionary that is able to convert some of the words with apostrophes back to their original form
    apostrophes = {
        "could n't" : "could not",
        "did n't" : "did not",
        "does n't" : "does not",
        "do n't" : "do not",
        "had n't" : "had not",
        "has n't" : "has not",
        "have n't" : "have not",
        "he'd" : "he would",
        "he'll" : "he will",
        "he's" : "he is",
        "i'd" : "I would",
        "i'd" : "I had",
        "i'll" : "I will",
        "i'm" : "I am",
        "is n't" : "is not",
        "it's" : "it is",
        "it'll":"it will",
        "i've" : "I have",
        "let's" : "let us",
        "might n't" : "might not",
        "must n't" : "must not",
        "sha" : "shall",
        "she'd" : "she would",
        "she'll" : "she will",
        "she's" : "she is",
        "should n't" : "should not",
        "that's" : "that is",
        "there's" : "there is",
        "they'd" : "they would",
        "they'll" : "they will",
        "they're" : "they are",
        "they've" : "they have",
        "we'd" : "we would",
        "we're" : "we are",
        "were n't" : "were not",
        "we've" : "we have",
        "what'll" : "what will",
        "what're" : "what are",
        "what's" : "what is",
        "what've" : "what have",
        "where's" : "where is",
        "who'd" : "who would",
        "who'll" : "who will",
        "who're" : "who are",
        "who's" : "who is",
        "who've" : "who have",
        "wo" : "will",
        "would n't" : "would not",
        "you'd" : "you would",
        "you'll" : "you will",
        "you're" : "you are",
        "you've" : "you have",
        "'re": " are",
        "was n't": "was not",
        "we'll":"we will",
        "did n't": "did not",
        "n't": "not",
    }

    #Call spacy to do some cleaning to the text
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
    #collection of stopwords to be removed
    stop = stopwords.words('english')
    all_=[]

    for review in reviews:
        #lowercase text
        lower_case = review.lower()
        #remove punctuation
        lower_case = lower_case.replace("."," . ")
        lower_case = ' '.join(word.strip(string.punctuation) for word in lower_case.split())
        #split text into words
        words = lower_case.split()
        #remove numbers
        words = [word for word in words if word.isalpha()]
        #correct some of the words with apostrophes (declared dictionary)
        split = [apostrophes[word] if word in apostrophes else word for word in words]
        #remove stopwords
        split = [word for word in split if word not in stop]
        #join the new words back to the text
        reformed = " ".join(split)
        doc = nlp(reformed)
        #lemmatization with spacey
        reformed = " ".join([token.lemma_ for token in doc])
        all_.append(reformed)

    df_cleaned = pd.DataFrame()
    df_cleaned['clean_reviews'] = all_
    return df_cleaned

def app():

    st.subheader("Test our trained model!")
    st.markdown("""Provide a review about a hotel (in text form) and check the predicted rating.
    Our trained model is based on the __Logistic Regression__ algorithm. This method applies the "one-vs-the-rest" multi-class strategy function to perform classification.
    The objective of this algorithm is to fit to the input, training data and return a "best fit" hyperplane that classifies them. Next step, the test data is provided to the classifier, to check what the "predicted" class is. """)
    user_input = st.text_area("Enter your review", "")
    submit_text = st.button('Submit Review')

    if(submit_text): #& len(user_input) > 0

        #processedText = cleanData(user_input)
        loaded_model = joblib.load('Material/Models/LogisticRegression_Model.pkl')
        loaded_tfidf = joblib.load('Material/Models/tfidfvectorizer.pkl')
        df = pd.DataFrame()
        df = pd.DataFrame([user_input], columns=['Review'])
        copied_Review = df['Review'].copy()
        X_cleaned = cleanData(copied_Review)

        new_testdata_tfidf= loaded_tfidf.transform([X_cleaned['clean_reviews'][0]])
        value = loaded_model.predict(new_testdata_tfidf)

        for i in range(5):
            cols = st.columns(5)

        for i in range(value[0]):
            cols[i].markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"> <span class="fa fa-star" style="color: orange";"></span>', unsafe_allow_html=True)

        for i in range(value[0],5):
            cols[i].markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"> <span class="fa fa-star""></span>', unsafe_allow_html=True)

    else:
        st.write("Please provide a review")
