#import three app files
import pred_rate1
import pred_rate2
import about

from PIL import Image

import streamlit as st

st.set_page_config(page_title='Rating Prediction App', page_icon=':smiley', layout="wide")

PAGES = {
    "About the assignment": about,
    "Rating prediction (file)": pred_rate1,
    "Rating prediction (dynamic)": pred_rate2,
}

st.sidebar.title("CEI 523: Data Science")

image = Image.open('Material/Images/stars.png')
st.sidebar.image(image, caption='Give a hotel review and predict the rating')

st.sidebar.header('Navigation Bar')
selection = st.sidebar.radio("Visit Page", list(PAGES.keys()))
page = PAGES[selection]
page.app()

st.sidebar.header('Assignment Description')
st.sidebar.markdown("""The current web app has been developed for the needs of the course: __*CEI 523 Data Science*__.
\n Our team (Group 3/Team 3) was responsible for developing a model which can predict the rating of a hotel based on a user review (text).\n
In the __"Rating prediction (file)"__ web-page, we present, step-by-step, our process for finding the best model. In this case, we utilize a known file.
This file has two columns: a "Review" column that contains users' comments on a hotel and another column "Rating" with
the respective rating (ranging from 1 to 5).
\n
In the __"Rating prediction (dynamic)"__ web-page, we test our models (as developed) with an unknown file, added by you!
Moreover, you are given the choice to configure the file as you like (e.g., replace unknown values with one of the Central Tendency measures, specify the range of the Rating column, etc.)\n
The input file needs to have at least two columns: a column that contains users' comments on a hotel and another column with the respective rating (with any range).
The column with users' comments (reviews) needs to include text (numbers, special characters may be included), while the column with the rating needs to include only numbers.
""")

st.sidebar.markdown("""__Team Members:__ Maria Hlia, Loukas Konstantinou""")
st.sidebar.markdown("""__Professor:__ Dr Andreas Christoforou""")
st.sidebar.markdown("""__Date:__ Winter Semester 2021""")

image = Image.open('Material/Images/cut.png')
st.sidebar.image(image, caption='Cyprus University of Technology')
