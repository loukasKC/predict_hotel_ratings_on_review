#pred_rate2.py

#import three libraries for dataframe, data manipulation
import streamlit as st
import pandas as pd
import numpy as np

#imports for using info()
import os
import io

#imports for graphs
import plotly
import plotly.express as px

#imports for wordcloud
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

#imports for NLP
import spacy
import nltk
from nltk.corpus import stopwords
import string

#imports for counting most frequent words in Review
from collections import Counter

#imports to prepare the data for the algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

#imports for models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import mord
from sklearn.neighbors import KNeighborsClassifier

#import for saving the models
import joblib

#imports for plotting
import seaborn as sns
from yellowbrick.classifier import ClassPredictionError

#method for adjusting the wordcloud layout
def black_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return("hsl(0,100%, 1%)")

#Method to clean the review column
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
    #return df_cleaned['clean_reviews']
    return df_cleaned

def app():

    #empty dataframe to insert the dataset (arrived by any way)
    df = pd.DataFrame()

    st.title("Rating prediction (dynamic)")
    st.header("Assignment description")
    st.markdown("""The current web app has been developed for the needs of the course: __*CEI 523 Data Science*__.
    \n Our team (Group 3) was responsible for developing a model which can predict the rating of a hotel based on a user review (text)
    \n In this web-page we present, step-by-step, our process for finding the best model.
    \n In this case, we utilize an unknown file which is added by you, the user!
    \n This file needs to have at least two columns: a column that contains users' comments on a hotel and another column with
    \n the respective rating (with any range).
    """)

    st.header("Step 1: Import your dataset")
    option = st.selectbox('How will you insert your dataset:', ('Upload a CSV/XLSX file', 'Use an API link', 'Test with Kaggle CSV file'))

    if option == 'Upload a CSV/XLSX file':
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            if(".xlsx" in uploaded_file.name):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                st.write(".xlsx")
            else:
                df = pd.read_csv(uploaded_file)
                st.write(".csv")

    elif option == 'Test with Kaggle CSV file':
        df = pd.read_csv("Material/Data/data_2.csv", sep=',',)

    elif option == 'Use an API link':
        form = st.form(key='my-form')
        name = form.text_input('Enter a dataset link')
        submit = form.form_submit_button('Submit')

        if submit:
            response = requests.get(name)
            st.write(response.headers['Content-Type'])
            if("json" in response.headers['Content-Type']):
                json = response.json()
                df = pd.DataFrame(json['response']['docs'])

    if not df.empty:

        st.markdown("__The input dataset__")
        st.write(df.head(4))
        column_datatype, column_data = st.columns(2)

        column_datatype.markdown("__Dataset structure__")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        column_datatype.text(s)

        column_data.markdown("__Dataset statistic description__")
        column_data.write(df.describe())

        st.markdown('#')
        df_official = pd.DataFrame()
        dataset_list = []

        st.header("Step 2: Select your columns")
        for col in df.columns:
            dataset_list.append(col)

        if 'form_submission' not in st.session_state:
            st.session_state['form_submission'] = False

        form = st.form(key='my-form')
        review_column = form.selectbox('Select the Review column', dataset_list)
        rating_column = form.selectbox('Select the Rating column', dataset_list)
        submit = form.form_submit_button('Submit')

        if(submit|st.session_state['form_submission']):
            st.session_state['form_submission'] = True
            df_official['Review'] = df[review_column].copy()
            df_official['Rating'] = df[rating_column].copy()


        if not df_official.empty:

            st.header("Step 3: Clean & transform the dataset")
            #review_column = df_official.iloc[:, 0]
            #rating_column = df_official.iloc[:, 1]
            numberRows, duplicatedRowsBoth, duplicatedRowsReview = st.columns(3)

            numberRows.markdown("__Dataframe presentation__")
            numberRows.write(df_official.head(4))
            numberRows.markdown("__Number of rows for both columns__")
            numberRows.write("Review (num. rows): " + str(df_official['Review'].count()))
            numberRows.write("Rating (num. rows): " + str(df_official['Rating'].count()))

            duplicatedRowsBoth.markdown("__Duplicated rows based on Review & Rating__")
            duplicatedRowsBoth.text("Duplicate rows (before removal): " + str(df_official.duplicated().sum()))
            duplicatedRowsBoth.text(df_official[df_official.duplicated(keep=False)])

            if(df_official.duplicated().sum()>0):
                df_official = df_official.drop_duplicates(subset=[df_official.columns[0], df_official.columns[1]], keep="first")
                df_official = df_official.reset_index(drop=True)

            duplicatedRowsBoth.markdown("__Duplicated rows based on Review & Rating__")
            duplicatedRowsBoth.text("Duplicate rows (after removal): " + str(df_official.duplicated().sum()))
            duplicatedRowsBoth.text(df_official[df_official.duplicated(keep=False)])

            duplicatedRowsReview.markdown("__Duplicated rows based on Review__")
            duplicatedRowsReview.text("Duplicate rows (before removal): " + str(df_official.duplicated(subset=['Review']).sum()))
            duplicatedRowsReview.text(df_official[df_official.duplicated(subset=['Review'], keep=False)])

            if(df_official.duplicated(subset=['Review']).sum()>0):
                df_official = df_official.drop_duplicates(subset=[df_official.columns[0]], keep="first")
                df_official = df_official.reset_index(drop=True)

            duplicatedRowsReview.markdown("__Duplicated rows based on Review__")
            duplicatedRowsReview.text("Duplicate rows (after removal): " + str(df_official.duplicated(subset=['Review']).sum()))
            duplicatedRowsReview.text(df_official[df_official.duplicated(subset=['Review'], keep=False)])

            st.markdown('#')

            st.markdown("__Information on outliers__")

            slideCol1, slideCol2, slideCol3 = st.columns(3)

            slideCol1.markdown("__Rating distribution__")
            slideCol1.write(df_official['Rating'].value_counts())

            min_rating = int(df_official['Rating'].min())
            max_rating = int(df_official['Rating'].max())

            slideCol2.markdown("__Adjust outliers__")
            values = slideCol2.slider('Select a range of values', min_rating, max_rating, (min_rating, max_rating))
            slideCol2.write('Values: ' + str(values[0]) + " & " + str(values[1]))
            submit_minmax = slideCol2.button('Submit min/max', key="minmax")

            if 'submit_minmax' not in st.session_state:
                st.session_state['submit_minmax'] = False

            if(submit_minmax | st.session_state['submit_minmax']):
                df_official = df_official[ ((df_official['Rating'] >= values[0]) & (df_official['Rating'] <= values[1])) |  (df_official['Rating'].isnull())]
                st.session_state['submit_minmax'] = True
                df_official = df_official.reset_index(drop=True)

                slideCol3.markdown("__New Rating distribution__")
                slideCol3.write(df_official['Rating'].value_counts())

                st.markdown("__Missing Values__")
                st.write("Missing values for Review: " + str(df_official['Review'].isna().sum()))

                if(df_official['Review'].isna().sum()>0):
                    df_official = df_official.dropna(subset=['Review'])
                    df_official = df_official.reset_index(drop=True)
                    st.write("Missing values for Review (after drop): " + str(df_official['Review'].isna().sum()))

                st.markdown('#')

                st.write("Missing values for Rating: " + str(df_official['Rating'].isna().sum()))

                #mean = df_official['Rating'].mean().astype(int)
                mean = df_official['Rating'].mean().astype(np.int64)
                median = df_official['Rating'].median().astype(np.int64)
                mode = df_official['Rating'].mode().astype(np.int64)

                if(mode.shape[0]>=1):
                    mode_list = [1]
                    for index, value in mode.items():
                        mode_list.append(value)
                    st.session_state['mode'] = st.selectbox('Select mode', mode_list)
                    mode = st.session_state['mode']

                st.write("Median: " +str(median))
                st.write("Mean: " + str(mean))
                st.write("Mode: ", mode)

                if 'mode' not in st.session_state:
                    st.session_state['mode'] = -1

                oldDF, makeChange, updatedDF = st.columns(3)

                oldDF.markdown("__Before replacing NaN values__")
                oldDF.write(df_official.describe())

                makeChange.markdown("__Replace the NaN values with:__")
                replace_choice = makeChange.radio("Replace the null values with:", ('Mode', 'Median', 'Mean'))

                if(replace_choice=="Mode"):
                    df_official['Rating'] = df_official['Rating'].fillna(mode)
                elif(replace_choice=="Median"):
                    df_official['Rating'] = df_official['Rating'].fillna(median)
                elif(replace_choice=="Mean"):
                    df_official['Rating'] = df_official['Rating'].fillna(mean)

                updatedDF.markdown("__After replacing NaN values__")
                updatedDF.write(df_official.describe())
                updatedDF.write("Missing values for Rating (after filling NaN): " + str(df_official['Rating'].isna().sum()))

                st.markdown('#')

                structureOldDF, structureNewDF = st.columns(2)

                structureOldDF.markdown("__Structure of Review & Rating (before)__")
                buffer = io.StringIO()
                df_official.info(buf=buffer)
                s = buffer.getvalue()
                structureOldDF.text(s)

                if df_official['Rating'].dtype == np.float64:
                    df_official['Rating'] = df_official['Rating'].astype(np.int64)
                    #df_official['Rating'] = df_official['Rating'].astype(np.int64)

                elif df_official['Rating'].dtype == np.object:
                    df_official['Rating'] = df_official['Rating'].astype(np.int64)

                structureNewDF.markdown("__Structure of Review & Rating (after)__")
                buffer = io.StringIO()
                df_official.info(buf=buffer)
                s = buffer.getvalue()
                structureNewDF.text(s)

                st.markdown('#')

                applyModels = st.button('Apply Models', key="applymodels")

                if(applyModels):

                    st.write(df_official.describe())

                    st.markdown('#')

                    st.subheader("Step 4: Prepare data for Modeling")

                    st.markdown("__Produced Dataframe__")
                    st.write(df_official.head())

                    newDataframeCol1, newDataframeCol2, newDataframeCol3 = st.columns(3)

                    newDataframeCol2.markdown('')
                    newDataframeCol3.markdown('#')
                    newDataframeCol3.markdown('#')
                    newDataframeCol3.markdown("__New Rating distribution__")
                    newDataframeCol3.write(df_official.describe())

                    #Generate a new df with two columns: the rating labels (1-5) and their respective counts
                    df2_values = df_official['Rating'].value_counts()
                    df2_label = df_official['Rating'].value_counts().index.tolist()
                    data_new = {'Ratings': df2_label, 'Frequency': df2_values}
                    df_rat_freq = pd.DataFrame(data_new)

                    fig = px.bar(df_rat_freq, x='Ratings', y='Frequency', color='Frequency', width=450)
                    newDataframeCol1.plotly_chart(fig)

                    st.markdown('#')

                    st.markdown("__Examine & process the Review column (text)__")
                    st.write("View all the words in the Review (pre-process)")

                    dfinal = pd.DataFrame()
                    dfinal['Rating'] = df_official['Rating'].copy()

                    #dataframe containing the cleaned review column
                    copied_Review = df_official['Review'].copy()
                    X_cleaned = cleanData(copied_Review)
                    dfinal['Review'] = X_cleaned['clean_reviews'].copy()
                    #dfinal['Review'] = dfinal['Review'].astype(str)

                    reviewHead_col1, reviewHead_col2 = st.columns(2)
                    reviewHead_col1.markdown("__Review before cleaning__")
                    reviewHead_col1.write(df_official['Review'].head(4))
                    reviewHead_col2.markdown("__Review after cleaning__")
                    reviewHead_col2.write(dfinal['Review'].head(4))

                    wordcloud_col1, wordcloud_col2 = st.columns(2)
                    wordcloud1 = WordCloud(
                                    scale=3,relative_scaling=1,width=7000,height=7000,
                                    max_words=500,colormap='RdYlGn',collocations=False,
                                    background_color='white',contour_color='white',).generate(' '.join(df_official['Review']))#the join function returns a text
                    wordcloud1.recolor(color_func = black_color_func)
                    plt.imshow(wordcloud1, interpolation='bilinear')
                    plt.axis("off")
                    plt.show()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    wordcloud_col1.pyplot()

                    wordcloud2 = WordCloud(
                                    scale=3,relative_scaling=1,width=7000,height=7000,
                                    max_words=500,colormap='RdYlGn',collocations=False,
                                    background_color='white',contour_color='white',).generate(' '.join(dfinal['Review']))#the join function returns a text
                    wordcloud2.recolor(color_func = black_color_func)
                    plt.imshow(wordcloud2, interpolation='bilinear')
                    plt.axis("off")
                    plt.show()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    wordcloud_col2.pyplot()

                    top10_col1, top10_col2 = st.columns(2)

                    top10_col1.markdown("__Top 10 words before cleaning__")
                    freq_words_list = Counter(" ".join(df_official["Review"]).split()).most_common(10)
                    words = [word for word, _ in freq_words_list]
                    counts = [counts for _, counts in freq_words_list]
                    fig1 = px.bar(freq_words_list, x = words, y = counts, color = counts, width=450)
                    top10_col1.plotly_chart(fig1)

                    top10_col2.markdown("__Top 10 words after cleaning__")
                    freq_words_list = Counter(" ".join(dfinal["Review"]).split()).most_common(10)
                    words = [word for word, _ in freq_words_list]
                    counts = [counts for _, counts in freq_words_list]
                    fig2 = px.bar(freq_words_list, x = words, y = counts, color = counts, width=450)
                    top10_col2.plotly_chart(fig2)

                    st.subheader("Step 5: Apply Models")

                    loaded_tfidf = joblib.load('Material/Models/tfidfvectorizer.pkl')
                    loaded_Linear = joblib.load('Material/Models/LinearSVC_Model.pkl')
                    loaded_LR = joblib.load('Material/Models/LogisticRegression_Model.pkl')
                    loaded_Logistic = joblib.load('Material/Models/LogisticIT_Model.pkl')
                    loaded_KNN = joblib.load('Material/Models/KNN_Model.pkl')

                    st.markdown('#')

                    X = dfinal['Review'].copy()
                    y = dfinal['Rating'].copy()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.15, random_state=3)
                    X_train_vectorized = loaded_tfidf.fit_transform(X_train)
                    X_test_vectorized = loaded_tfidf.transform(X_test)

                    st.write("Number of unique words for training:", X_train_vectorized.shape)
                    st.write("Number of unique words for testing:", X_test_vectorized.shape)
                    labels=[1,2,3,4,5]

                    st.subheader("Multi-Classification Algorithm: Linear Support Vector Clustering")
                    loaded_Linear.fit(X_train_vectorized, y_train)
                    y_pred = loaded_Linear.predict(X_test_vectorized)

                    st.markdown('#')

                    st.subheader("Multi-Classification Algorithm: Logistic Regression")
                    loaded_LR.fit(X_train_vectorized, y_train)
                    y_predLR = loaded_LR.predict(X_test_vectorized)

                    st.markdown('#')

                    st.subheader("mord package: LogisticIT")
                    loaded_Logistic.fit(X_train_vectorized, y_train)
                    mord_ypredLR = loaded_Logistic.predict(X_test_vectorized)

                    st.markdown('#')

                    st.subheader("Multi-Classification Algorithm: KNeighbors")
                    loaded_KNN.fit(X_train_vectorized, y_train)
                    y_predknn = loaded_KNN.predict(X_test_vectorized)

                    st.markdown('#')

                    st.subheader("Step 4: Compare Models")

                    st.markdown("__Accuracy Score:__ Accuracy based on predicted y and true y (how much the two sets correspond)")
                    accuracyLinear, accuracyRegression = st.columns(2)
                    accuracyMode, accuracyKNN = st.columns(2)

                    accuracyLinear.text("Linear SVC accuracy score: {:.2}".format(accuracy_score(y_test, y_pred)))
                    accuracyRegression.text("Logistic Regression accuracy score: {:.2}".format(accuracy_score(y_test, y_predLR)))
                    accuracyMode.text("Mord accuracy score: {:.2}".format(accuracy_score(y_test, mord_ypredLR)))
                    accuracyKNN.text("KNN accuracy score: {:.2}".format(accuracy_score(y_test, y_predknn)))

                    st.markdown('#')

                    st.markdown("__Classification Report:__ ")

                    classReportLinear, classReportRegression = st.columns(2)
                    classReportMode, classReportKNN = st.columns(2)

                    #Plot Classification for Linear
                    classReportLinear.write("Linear SVC")
                    class_report_linear = classification_report(y_test, y_pred, output_dict=True)
                    classification_report_linear_plot = plt.figure()
                    sns.heatmap(pd.DataFrame(class_report_linear).iloc[:-1, :].T, annot=True, fmt='.2g')
                    classReportLinear.pyplot(classification_report_linear_plot)

                    #Plot Classification for Logistic
                    classReportRegression.write("Logistic Regression")
                    class_report_logistic = classification_report(y_test, y_predLR, output_dict=True)
                    classification_report_logistic_plot = plt.figure()
                    sns.heatmap(pd.DataFrame(class_report_logistic).iloc[:-1, :].T, annot=True, fmt='.2g')
                    classReportRegression.pyplot(classification_report_logistic_plot)

                    #Plot Classification for Mord
                    classReportMode.write("LogisticIT")
                    class_report_mord = classification_report(y_test, mord_ypredLR, output_dict=True)
                    classification_report_mord_plot = plt.figure()
                    sns.heatmap(pd.DataFrame(class_report_mord).iloc[:-1, :].T, annot=True, fmt='.2g')#iloc[:-1, :]
                    classReportMode.pyplot(classification_report_mord_plot)

                    #Plot Classification for KNN
                    classReportKNN.write("KNN")
                    class_report_knn = classification_report(y_test, y_predknn, output_dict=True)
                    classification_report_knn_plot = plt.figure()
                    sns.heatmap(pd.DataFrame(class_report_knn).iloc[:-1, :].T, annot=True, fmt='.2g')
                    classReportKNN.pyplot(classification_report_knn_plot)

                    st.markdown('#')

                    st.markdown("__Confusion Matrix:__ ")
                    confuReportLinear, confuReportRegression = st.columns(2)
                    confuReportMode, confuReportKNN = st.columns(2)

                    #Plot Matrix for Linear
                    confuReportLinear.write("Linear SVC")
                    matrixSVC = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
                    confusion_matrix_plot = plt.figure()
                    sns.heatmap(matrixSVC , annot=True , xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], fmt='.2g')
                    plt.ylabel("True Values")
                    plt.xlabel("Predicted Values")
                    confuReportLinear.pyplot(confusion_matrix_plot)

                    #Plot Matrix for Regression
                    confuReportRegression.write("Logistic Regression")
                    matrixLR = confusion_matrix(y_test, y_predLR, labels=labels, normalize='true')
                    confusion_matrix_plot_LR = plt.figure()
                    sns.heatmap(matrixLR , annot=True , xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], fmt='.2g')
                    plt.ylabel("True Values")
                    plt.xlabel("Predicted Values")
                    confuReportRegression.pyplot(confusion_matrix_plot_LR)

                    #Plot Matrix for Mord
                    confuReportMode.write("LogisticIT")
                    matrixMord = confusion_matrix(y_test, mord_ypredLR, labels=labels, normalize='true')
                    confusion_matrix_plot_Mord = plt.figure()
                    sns.heatmap(matrixMord , annot=True , xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], fmt='.2g')
                    plt.ylabel("True Values")
                    plt.xlabel("Predicted Values")
                    confuReportMode.pyplot(confusion_matrix_plot_Mord)

                    #Plot Matrix for KNN
                    confuReportKNN.write("KNN")
                    matrixKNN = confusion_matrix(y_test, y_predknn, labels=labels, normalize='true')
                    confusion_matrix_plot_Knn = plt.figure()
                    sns.heatmap(matrixKNN , annot=True , xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5], fmt='.2g')
                    plt.ylabel("True Values")
                    plt.xlabel("Predicted Values")
                    confuReportKNN.pyplot(confusion_matrix_plot_Knn)

                    st.markdown('#')

                    st.markdown("__Classification Prediction Error:__ Extension of confusion matrix")
                    CPEReportLinear, CPEReportRegression = st.columns(2)
                    CPEReportMode, CPEReportKNN = st.columns(2)

                    # Cl.Pr.Er. for linear
                    CPEReportLinear.write("Linear SVC")
                    visualizerLinear = ClassPredictionError(loaded_Linear)
                    visualizerErrorsLinear = plt.figure()
                    visualizerLinear.fit(X_train_vectorized, y_train)
                    visualizerLinear.score(X_test_vectorized, y_test)
                    CPEReportLinear.pyplot(visualizerErrorsLinear)

                    # Cl.Pr.Er. for logistic
                    CPEReportRegression.write("Logistic Regression")
                    visualizerRegression = ClassPredictionError(loaded_LR)
                    visualizerErrorsRegression = plt.figure()
                    visualizerRegression.fit(X_train_vectorized, y_train)
                    visualizerRegression.score(X_test_vectorized, y_test)
                    CPEReportRegression.pyplot(visualizerErrorsRegression)

                    # Cl.Pr.Er. for mord
                    CPEReportMode.write("LogisticIT")
                    visualizerMode = ClassPredictionError(loaded_Logistic, support=True, force_model=True)
                    visualizerErrorsMode = plt.figure()
                    visualizerMode.fit(X_train_vectorized, y_train)
                    visualizerMode.score(X_test_vectorized, y_test)
                    CPEReportMode.pyplot(visualizerErrorsMode)

                    # Cl.Pr.Er. for KNN
                    CPEReportKNN.write("KNN")
                    visualizerKNN = ClassPredictionError(loaded_KNN)
                    visualizerErrorsKNN = plt.figure()
                    visualizerKNN.fit(X_train_vectorized, y_train)
                    visualizerKNN.score(X_test_vectorized, y_test)
                    CPEReportKNN.pyplot(visualizerErrorsKNN)

                    st.markdown('#')

                    st.subheader("Cross Validation Score")

                    col1explanation, col2graph = st.columns(2)

                    names = ["LinearSVC", "Logistic Regression", "LogisticIT", "KNN"]
                    results = []
                    from sklearn import model_selection

                    kfoldLinear = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
                    cv_resultsLinear = model_selection.cross_val_score(loaded_Linear, X_train_vectorized, y_train, cv=kfoldLinear, scoring='accuracy')
                    results.append(cv_resultsLinear)

                    kfoldLogistic = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
                    cv_resultsLogistic = model_selection.cross_val_score(loaded_LR, X_train_vectorized, y_train, cv=kfoldLogistic, scoring='accuracy')
                    results.append(cv_resultsLogistic)

                    kfoldMord = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
                    cv_resultsMord = model_selection.cross_val_score(loaded_Logistic, X_train_vectorized, y_train, cv=kfoldMord, scoring='accuracy')
                    results.append(cv_resultsMord)

                    kfoldKNN = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
                    cv_resultsLinear = model_selection.cross_val_score(loaded_KNN, X_train_vectorized, y_train, cv=kfoldKNN, scoring='accuracy')
                    results.append(cv_resultsLinear)

                    col1explanation.markdown("""
                    Examine the spread of the accuracy scores, across each cross validation fold for each algorithm:
                    A 10-fold cross validation procedure, used to evaluate each algorithm.
                    This procedure is configured with the same random seed to ensure that the same splits to the training data are
                    performed and that each algorithm is evaluated in the exact same way.
                    [[source]](https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/)
                    """)

                    tickvalues = range(1,5)
                    fig = plt.figure()
                    fig.suptitle('Algorithm Comparison')
                    ax = fig.add_subplot(111)
                    plt.violinplot(results)
                    plt.ylim(np.min(results)-0.5, np.max(results)+0.5)
                    plt.xticks(ticks=tickvalues,labels=names, rotation = 'horizontal')
                    plt.show()
                    col2graph.pyplot(fig)

    else:
        st.write("No dataset imported")
