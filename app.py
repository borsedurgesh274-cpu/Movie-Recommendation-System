import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "movies.csv")
    df = pd.read_csv(file_path)
    return df

movies = load_data()

# ---------------------------------
# Recommendation Function
# ---------------------------------
def recommend(movie_name):

    cv = CountVectorizer()
    matrix = cv.fit_transform(movies['genres'])

    similarity = cosine_similarity(matrix)

    movie_index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")

    for movie in recommendations:
        st.write("👉", movie)
