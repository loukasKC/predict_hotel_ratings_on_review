#pred_rate1.py

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
nltk.download('stopwords')
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

    #empty dataframe to store the csv
    df = pd.DataFrame()

    st.title("Rating prediction (file)")

    st.subheader("Introduction")
    with st.expander('Show More'):
        st.markdown("""
        For solving this predictive modeling problem (develop a model using historical data, for making a prediction on new unknown data) and finding the best
        function approximation (__Y = f(X)+ C__), we decided to go with the classification route. [[source]](https://builtin.com/data-science/supervised-learning-python) Classification predictive modeling attempts to predict categorical class labels, that are discrete and unordered. [[here]](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)\n
        As seen below, the labels for prediction are limited and discrete, ranging from 1 star (very negative) to 5 stars (very positive). We can treat each number as a separate, unique class and therefore, treat this as a classification task. [[here]](https://towardsdatascience.com/1-to-5-star-ratings-classification-or-regression-b0462708a4df)\n

        However, we need to highlight the fact that each category is an integer value (1 corresponds to the first class, 2 to the second class and so on) and there's no need for integer encoding.
        These 5 integer values have a natural ordered relationship between them and machine learning algorithms may be able to understand and exploit this relationship.
        [[source]](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
        """)

    st.subheader("Step 1: Open the dataset")
    df = pd.read_csv("Material/Data/data.csv")
    with st.expander('Show More'):
        st.markdown("""
        The CSV file is opened and stored as a dataframe. As seen below, it has two columns:\n
        __Review:__ and __Rating__""")
        st.write(df.head(4))

    st.subheader("Step 2: Clean & transform the data")
    with st.expander('Show More'):
        st.markdown("""
        The data cleaning and transformation process is the following:

            1. Check for duplicates (two conditions)
                a. Duplicated rows for both columns (same Review & Rating rows)
                b. Duplicated rows for Review column (possible case of rows with same Review but different Rating). In this case, we need to keep only one row\n
            2. Check the structure of columns
                a. Review column needs to be an object (mixed numeric & non-numeric values)
                b. Rating column needs to be an integer64 (integer values from 1 to 5)\n
            3. Information on statistics & outliers (Rating)
                a. Verify there are no outliers (values outside the range 1-5)\n
            4. Check for missing data (both columns need to have 0 NaN values)\n
            5. Examine & process the Review column (text that needs to be cleaned)
                a. Lowercase words
                b. Remove punctuation
                c. Other cleaning (e.g., remove numbers, apostrophes)
                d. Remove stopwords
                e. Lemmatize words
        """)

    numberRows, duplicatedRows, missingRows = st.columns(3)

    numberRows.markdown("__Check the number of rows for both columns__")
    numberRows.text("Review (num. rows): " + str(df['Review'].count()))
    numberRows.text("Rating (num. rows): " + str(df['Rating'].count()))
    numberRows.caption("We have same number of rows for both columns")

    duplicatedRows.markdown("__Check duplicated rows for both conditions__")
    duplicatedRows.text("Duplicate rows (Review & Rating): " + str(df.duplicated().sum()))
    duplicatedRows.text("Duplicate rows (Review): " + str(df.duplicated(subset=['Review']).sum()))
    duplicatedRows.caption("We have no duplicate values for both conditions")

    missingRows.markdown("__Check for missing data (both columns)__")
    missingRows.text("Missing values (Review): " + str(df['Review'].isna().sum()))
    missingRows.text("Missing values (Rating): " + str(df['Rating'].isna().sum()))
    missingRows.caption("Number of missing values is 0 for both")

    st.markdown('#')

    st.markdown("__Check the structure of columns__")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.caption("Verified column types")

    st.markdown('#')

    st.markdown("__Check the statistics & outliers of column Rating (int)__")

    #Generate a new df with two columns: the rating labels (1-5) and their respective counts
    df2_values = df['Rating'].value_counts()
    df2_label = df['Rating'].value_counts().index.tolist()
    data_new = {'Ratings': df2_label, 'Frequency': df2_values}
    df_rat_freq = pd.DataFrame(data_new)

    fig = px.bar(df_rat_freq, x='Ratings', y='Frequency', color='Frequency', width=600)
    rate_col1, rate_col2, col3 = st.columns(3)
    rate_col1.plotly_chart(fig)
    rate_col2.markdown('')
    col3.markdown('#')
    col3.markdown('#')
    col3.write(df.describe())
    col3.caption("Review statistics chart")

    st.markdown('#')

    st.markdown("__Examine & process the Review column (text)__")
    st.write("View all the words in the Review (pre-process)")

    st.write("Cleaning process of Review")
    with st.expander('Show More'):
        st.markdown("""
        -   __Lowercase words__: We lowercase all words in order to perform basic functions to all of them and not miss any.
        We checked that stopwords, lemmatization, etc. are not performed on capitalized letters, therefore we need to bring
        all words to the same lowercase form

        -   __Remove punctuation__: According to some articles we examined, it is claimed that punctuation doesn't add much
        meaning when searching for a topic or attempting to ascertain sentiment. Therefore, we decided to remove it from text
        [[source]](https://www.nicholasrenotte.com/how-to-build-a-sentiment-analyser-for-yelp-reviews-in-python/),
        [[source]](https://medium.com/analytics-vidhya/predicting-the-ratings-of-reviews-of-a-hotel-using-machine-learning-bd756e6a9b9b),
        [[source]](https://towardsdatascience.com/nlp-in-python-data-cleaning-6313a404a470)

        - __Other cleaning (e.g., remove numbers)__: We decided to remove numbers from our reviews, since we want to focus
        only on the text that users provided

        -   __Remove punctuation (extra)__: Along with punctuation, we discovered some words with apostrophes that needed to be
        converted in their original form (e.g., "could n't" needs to be converted to "could not"). So, we declared a dictionary that
        corrects any instances of words with apostrophes, back to their original form

        -   __Remove stopwords__: Moreover, we decided to remove english stopwords (from the NLTK package), since they are considered
        as noise in our data
        [[source]](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)

        -   __Lemmatization__: Convert different forms of the same word back to its root (e.g., relaxing -> relax)
        """)

    #dataframe containing the cleaned review column
    copied_Review = df['Review'].copy()
    #X_cleaned = cleanData(copied_Review)

    #dfinal = pd.DataFrame()
    #dfinal['Review'] = X_cleaned['clean_reviews'].copy()
    #dfinal['Rating'] = df['Rating'].copy()

    #dfinal.to_csv("cleaned.csv", sep='\t', encoding='utf-8',index=False)

    #uncomment the 4 above Lines
    dfinal = pd.read_csv("Material/Data/cleaned.csv")#remove this
    dfinal['Rating'] = df['Rating'].copy()#remove this

    X = dfinal['Review'].copy()
    y = dfinal['Rating'].copy()

    #compare words from old df (raw text) and new df (cleaned text)
    st.markdown("__For resource purposes, we already the cleaned text is loaded from a CSV file__")
    reviewHead_col1, reviewHead_col2 = st.columns(2)
    reviewHead_col1.markdown("__Review before cleaning__")
    reviewHead_col1.write(df['Review'].head(4))
    reviewHead_col2.markdown("__Review after cleaning__")
    reviewHead_col2.write(dfinal['Review'].head(4))

    wordcloud_col1, wordcloud_col2 = st.columns(2)
    wordcloud_col1.markdown("__Top 500 words before cleaning__")
    mask = np.array(Image.open('Material/Images/hotel.jpg'))
    wordcloud1 = WordCloud(
    #background_color='white', mask=mask, mode='RGB',width=1000, max_words=1000, height=1000,random_state=1, contour_width=1, contour_color='steelblue'
                    scale=3,relative_scaling=1,width=7000,height=7000,
                    max_words=500,colormap='RdYlGn',collocations=False,mask=mask,
                    background_color='white',contour_color='white',).generate(' '.join(df['Review']))#the join function returns a text
    wordcloud1.recolor(color_func = black_color_func)
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    wordcloud_col1.pyplot()

    wordcloud_col2.markdown("__Top 500 words after cleaning__")
    mask = np.array(Image.open('Material/Images/hotel.jpg'))
    wordcloud2 = WordCloud(
    #background_color='white', mask=mask, mode='RGB',width=1000, max_words=1000, height=1000,random_state=1, contour_width=1, contour_color='steelblue'
                    scale=3,relative_scaling=1,width=7000,height=7000,
                    max_words=500,colormap='inferno',collocations=False,mask=mask,
                    background_color='white',contour_color='white',).generate(' '.join(dfinal['Review']))#the join function returns a text
    #wordcloud2.recolor(color_func = black_color_func)
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    wordcloud_col2.pyplot()

    top10_col1, top10_col2 = st.columns(2)

    top10_col1.markdown("__Top 10 words before cleaning__")
    freq_words_list = Counter(" ".join(df["Review"]).split()).most_common(10)
    words = [word for word, _ in freq_words_list]
    counts = [counts for _, counts in freq_words_list]
    fig = px.bar(freq_words_list, x = words, y = counts, color = counts, width=450)
    top10_col1.plotly_chart(fig)

    top10_col2.markdown("__Top 10 words after cleaning__")
    freq_words_list = Counter(" ".join(dfinal["Review"]).split()).most_common(10)
    words = [word for word, _ in freq_words_list]
    counts = [counts for _, counts in freq_words_list]
    fig = px.bar(freq_words_list, x = words, y = counts, color = counts, width=450)
    top10_col2.plotly_chart(fig)

    st.markdown('#')

    st.subheader("Step 3: Apply Models")
    with st.expander('Show More'):
        st.markdown("""
        __Text Vectorization:__\n Before looking at the models, it is important to convert the text data into numeric data (numerical feature vectors)\n. Since, computers can only understand numbers and not text,\n with this conversion,
        our machine learning models will be able to leverage linear algebra and capture potential patterns of the words\n and proceed with the rating prediction.\n In this assignment, we are employing the __Tfidfvectorizer__.\n\n__Tfidfvectorizer__:  a method for converting textual data into vectors, since models can process only numerical data. TfidfVectorizer does three steps at once:\n -   Counts the term frequency (TF), limits vocabulary size, removes stopwords\n -   Computes the IDF (Inverse Document Frequency) values of the words, based on their appearance in each document (the lower the IDF value, the less unique it is to any document)\n -   Computes the TFIDF scores by multiplying the TF with the IDF. Some words appear frequently in one document and less in other documents and are treated as more insightful by this method (common words are penalized)\n In our case, based on experimentation and testing, we decided on the following TfidfVectorizer parameters:
        -   stop_words = english - (removal of most common words by sklearn)
        -   ngram_range = (1, 2) - (take phrases between 1/unigrams and 2 words/bigrams)
        -   max_df = 0.95 -(ignore phrases that are in 95% of reviews)
        -   min_df = 5 - (ignore phrases that are in fewer than 5 reviews)
        -   sublinear_tf - True (replaces term frequency with log(term frequency), normalises bias against lengthy vs short docs)
        """)

        st.markdown('#')
        st.markdown("__Before the operationalization__:")
        image = Image.open("/Material/Images/tfidfbefore.png")
        st.image(image, width=800)
        st.markdown("__After the operationalization__:")
        st.image("Material/Images/tfidfafter.png", width=800)

    tfidf = TfidfVectorizer(lowercase=True, strip_accents="ascii", analyzer='word', stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=5, sublinear_tf=True)#, ,

    st.markdown('#')

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.15, random_state=3)
    X_train_vectorized = tfidf.fit_transform(X_train)
    X_test_vectorized = tfidf.transform(X_test)

    st.write("Number of unique words for training:", X_train_vectorized.shape)
    st.write("Number of unique words for testing:", X_test_vectorized.shape)
    labels=[1,2,3,4,5]

    st.subheader("Multi-Classification Algorithm: Linear Support Vector Clustering")
    with st.expander('Show More'):
        st.markdown("""
         LinearSVC algorithm implements the "one-vs-the-rest" multi-class strategy and trains n_class models.
         Moreover, it performs well on on range of text classification tasks.
        [[source]](https://www.tutorialspoint.com/scikit_learn/scikit_learn_support_vector_machines.htm)
        [[source]](https://medium.com/@manoveg/multi-class-text-classification-with-probability-prediction-for-each-class-using-linearsvc-in-289189fbb100)\n
        __Operationalization:__\n
        - class weight (automatically adjust weights inversely proportional to class frequencies): balanced
        - C (how much you want to avoid misclassifying each training example): 1
        """)
        st.image("Material/Images/linear_c.png", width=800)
        st.caption("No need to change the default value")
        #error = []
        #for i in range(1, 40):
            #clf = LinearSVC(C=i)
            #clf.fit(X_train_vectorized, y_train)
            #pred_i = clf.predict(X_test_vectorized)
            #error.append(np.mean(pred_i != y_test))

        #plt.figure(figsize=(10, 4))
        #plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
        #plt.xlabel('C value')
        #plt.title('Error Rate C')
        #plt.ylabel('Mean Error')
        #st.pyplot()
    clf = LinearSVC(C=1, class_weight='balanced')
    clf.fit(X_train_vectorized, y_train)
    y_pred = clf.predict(X_test_vectorized)

    st.markdown('#')
    st.subheader("Multi-Classification Algorithm: Logistic Regression")
    with st.expander('Show More'):
        st.markdown("""
        A discriminative classification model that uses the one-vs-rest heuristic method, to find the label (splits multi-class dataset into a multiple binary classification problem).
        [[source]](https://www.mastersindatascience.org/learning/introduction-to-machine-learning-algorithms/logistic-regression)
        [[source]](https://www.tutorialspoint.com/scikit_learn/scikit_learn_support_vector_machines.htm)\n
        __Operationalization:__\n
        - class weight (weight inversely proportional to class frequencies): balanced
        - multi class (binary problem is fit for each label): ovr
        - solver (algorithm for optimization): liblinear (increased accuracy to 0.63)
        """)
        #error = []
        #for i in range(1, 40):
            #clf = LogisticRegression(C=i)
            #clf.fit(X_train_vectorized, y_train)
            #pred_i = clf.predict(X_test_vectorized)
            #error.append(np.mean(pred_i != y_test))
        #plt.figure(figsize=(10, 4))
        #plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
        #plt.xlabel('C value')
        #plt.title('Error Rate C')
        #plt.ylabel('Mean Error')
        #st.pyplot()
    clf_benchmark = LogisticRegression(class_weight='balanced', multi_class='ovr', solver= 'liblinear', C=1)
    clf_benchmark.fit(X_train_vectorized, y_train)
    y_predLR = clf_benchmark.predict(X_test_vectorized)

    st.markdown('#')

    st.subheader("mord package: LogisticIT")
    with st.expander('Show More'):
        st.markdown("""
        mord is a Python package that implements Ordinal Regression methods, based on the scikit-learn API.
        This package can be used extensively, in cases where the goal is to predict a discrete and ordered label.
        In our case, we are employing the __LogisticIT__, a classifier which implements the ordinal logistic model.
        [[source]](https://pythonhosted.org/mord/reference.html#mord.MulticlassLogistic)\n
        __Operationalization:__\n
        - alpha (learning parameter): 1
        """)
        st.image("Material/Images/mord_alpha.png", width=800)
        st.caption("No need to change the default value")
    #error = []
    #for i in range(1, 20):
        #mordLR = mord.LogisticAT(alpha=i)
        #mordLR.fit(X_train_vectorized, y_train)
        #pred_i = mordLR.predict(X_test_vectorized)
        #error.append(np.mean(pred_i != y_test))

    #plt.figure(figsize=(10, 4))
    #plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
    #plt.xlabel('alpha value')
    #plt.title('Error Rate alpha')
    #plt.ylabel('Mean Error')
    #st.pyplot()
    mordLR = mord.LogisticAT(alpha=1.0)
    mordLR.fit(X_train_vectorized, y_train)
    mord_ypredLR = mordLR.predict(X_test_vectorized)

    st.markdown('#')

    st.subheader("Multi-Classification Algorithm: KNeighbors")
    with st.expander('Show More'):
        st.markdown("""
         KNN (k-nearest neighbors) is a simple classification algorithm that does not depend on the structure of the data.
         Classifies a new example by measuring the distance between its k nearest n_neighbors
         The distance between two examples can be the euclidean distance between their feature vectors.
         The majority class among the k nearest neighbors is taken as the class for the new/given example.
        [[source]](https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/)
        [[source]](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn)\n
        __Operationalization:__\n
        - n_neighbors (number of neighbors to use): 5 (default)
        - leaf_size (weight function used in prediction): 30 (default)
        """)
        st.image("Material/Images/knn_neighbors.png", width=800)
        st.caption("Optimal n-neighbors number: 21")
        st.image("Material/Images/knn_leaf.png", width=800)
        st.caption("No change in leaf size number")
        #error = []
        #for i in range(0, 50):
            #knn_model = KNeighborsClassifier(n_neighbors=21, leaf_size=1)
            #knn_model.fit(X_train_vectorized, y_train)
            #pred_i = knn_model.predict(X_test_vectorized)
            #error.append(np.mean(pred_i != y_test))

        #plt.figure(figsize=(12, 6))
        #plt.plot(range(0, 50), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
        #plt.title('Error Rate Leaf Size Value')
        #plt.xlabel('Leaf Size Value')
        #plt.ylabel('Mean Error')
        #st.pyplot()
    knn_model = KNeighborsClassifier(n_neighbors=21)
    knn_model.fit(X_train_vectorized, y_train)
    y_predknn = knn_model.predict(X_test_vectorized)

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
    with st.expander('Show More'):
        st.markdown("""
        - Precision: how precise/accurate the models are (out of predicted positive, how many are actual positive)
        - Recall: how many of the actual positives, the models capture through labeling it as positive
        - F1: balance between Precision and Recall
        """)

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
    visualizerLinear = ClassPredictionError(clf)
    visualizerErrorsLinear = plt.figure()
    visualizerLinear.fit(X_train_vectorized, y_train)
    visualizerLinear.score(X_test_vectorized, y_test)
    CPEReportLinear.pyplot(visualizerErrorsLinear)

    # Cl.Pr.Er. for logistic
    CPEReportRegression.write("Logistic Regression")
    visualizerRegression = ClassPredictionError(clf_benchmark)
    visualizerErrorsRegression = plt.figure()
    visualizerRegression.fit(X_train_vectorized, y_train)
    visualizerRegression.score(X_test_vectorized, y_test)
    CPEReportRegression.pyplot(visualizerErrorsRegression)

    # Cl.Pr.Er. for mord
    CPEReportMode.write("LogisticIT")
    visualizerMode = ClassPredictionError(mordLR, support=True, force_model=True)
    visualizerErrorsMode = plt.figure()
    visualizerMode.fit(X_train_vectorized, y_train)
    visualizerMode.score(X_test_vectorized, y_test)
    CPEReportMode.pyplot(visualizerErrorsMode)

    # Cl.Pr.Er. for KNN
    CPEReportKNN.write("KNN")
    visualizerKNN = ClassPredictionError(knn_model)
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
    cv_resultsLinear = model_selection.cross_val_score(clf, X_train_vectorized, y_train, cv=kfoldLinear, scoring='accuracy')
    results.append(cv_resultsLinear)

    kfoldLogistic = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    cv_resultsLogistic = model_selection.cross_val_score(clf_benchmark, X_train_vectorized, y_train, cv=kfoldLogistic, scoring='accuracy')
    results.append(cv_resultsLogistic)

    kfoldMord = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    cv_resultsMord = model_selection.cross_val_score(mordLR, X_train_vectorized, y_train, cv=kfoldMord, scoring='accuracy')
    results.append(cv_resultsMord)

    kfoldKNN = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    cv_resultsLinear = model_selection.cross_val_score(knn_model, X_train_vectorized, y_train, cv=kfoldKNN, scoring='accuracy')
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
    plt.ylim(0.4, 0.8)
    plt.xticks(ticks=tickvalues,labels=names, rotation = 'horizontal')
    plt.show()
    col2graph.pyplot(fig)

    st.markdown('#')

    st.subheader("Some examples:")
    examplecol1, examplecol2, examplecol3 = st.columns(3)

    examplecol1.markdown("""
    The rooms were clean, very comfortable, and the staff was amazing.
    They went over and beyond to help make our stay enjoyable. I highly recommend this hotel for anyone visiting downtown [5]
    """)

    review1  = "The rooms were clean, very comfortable, and the staff was amazing. They went over and beyond to help make our stay enjoyable. I highly recommend this hotel for anyone visiting downtown"
    new_testdata_tfidf= tfidf.transform([review1])

    examplecol1.text("Linear SVC prediction: " + str(clf.predict(new_testdata_tfidf)))
    examplecol1.text("Logistic Regression prediction: " + str(clf_benchmark.predict(new_testdata_tfidf)))
    examplecol1.text("LogisticIT prediction: " + str(mordLR.predict(new_testdata_tfidf)))
    examplecol1.text("KNN prediction: " + str(knn_model.predict(new_testdata_tfidf)))


    examplecol2.markdown("""
    The rooms stink. They are not clean. I requested a non smoking room and both rooms smelled like smoke. Bathrooms were gross and bugs were everywhere! !
    The door did not seem secure and was not evened out so bugs got in easily. The second room was full of gnats. [1]
    """)

    review2  = "The rooms stink. They are not clean. I requested a non smoking room and both rooms smelled like smoke. Bathrooms were gross and bugs were everywhere! ! The door did not seem secure and was not evened out so bugs got in easily. The second room was full of gnats."
    new_testdata_tfidf= tfidf.transform([review2])

    examplecol2.text("Linear SVC prediction: " + str(clf.predict(new_testdata_tfidf)))
    examplecol2.text("Logistic Regression prediction: " + str(clf_benchmark.predict(new_testdata_tfidf)))
    examplecol2.text("LogisticIT prediction: " + str(mordLR.predict(new_testdata_tfidf)))
    examplecol2.text("KNN prediction: " + str(knn_model.predict(new_testdata_tfidf)))


    examplecol3.markdown("""
    Hotel has very good price, it has very good location inside city centre and very easily reachable to all locations.
    There was very small room only bed and one chair and you should stay only for a night sleep. [3]
    """)

    review3  = "Hotel has very good price, it has very good location inside city centre and very easily reachable to all locations. There was very small room only bed and one chair and you should stay only for a night sleep."
    new_testdata_tfidf= tfidf.transform([review3])

    examplecol3.text("Linear SVC prediction: " + str(clf.predict(new_testdata_tfidf)))
    examplecol3.text("Logistic Regression prediction: " + str(clf_benchmark.predict(new_testdata_tfidf)))
    examplecol3.text("LogisticIT prediction: " + str(mordLR.predict(new_testdata_tfidf)))
    examplecol3.text("KNN prediction: " + str(knn_model.predict(new_testdata_tfidf)))

    st.markdown('#')

    st.subheader("What about other studies?")

    st.markdown("""
    __Study 1:__ Predict hotel ratings on reviews\n
    __Model:__ Sequential Model\n
    __Accuracy:__ 0.59\n
    __Source:__ [[source]](https://medium.com/analytics-vidhya/predicting-the-ratings-of-reviews-of-a-hotel-using-machine-learning-bd756e6a9b9b)
    """)

    st.markdown('#')

    st.markdown("""
    __Study 2:__ Predict Yelp Stars from Reviews\n
    __Model:__ LinearSVC\n
    __Accuracy:__ 0.62\n
    __Source:__ [[source]](https://www.developintelligence.com/blog/2017/03/predicting-yelp-star-ratings-review-text-python/)
    """)

    st.markdown('#')

    st.markdown("""
    __Study 3:__ Predict Yelp Stars from Reviews (Sentiment Classification: positive/negative)\n
    __Model:__ Logistic Regression\n
    __Accuracy:__ 0.95\n
    __Source:__ [[source]](https://towardsdatascience.com/sentiment-classification-with-logistic-regression-analyzing-yelp-reviews-3981678c3b44)
    """)

    st.markdown('#')

    #Save the models

    joblib_model_LinearSVC = "Material/Models/LinearSVC_Model.pkl"
    joblib.dump(clf, joblib_model_LinearSVC)

    joblib_model_LogisticRegression = "Material/Models/LogisticRegression_Model.pkl"
    joblib.dump(clf_benchmark, joblib_model_LogisticRegression)

    joblib_model_LogisticIT = "Material/Models/LogisticIT_Model.pkl"
    joblib.dump(mordLR, joblib_model_LogisticIT)

    joblib_model_KNN = "Material/Models/KNN_Model.pkl"
    joblib.dump(knn_model, joblib_model_KNN)

    joblib_vectorizer  = "Material/Models/tfidfvectorizer.pkl"
    joblib.dump(tfidf, joblib_vectorizer)
