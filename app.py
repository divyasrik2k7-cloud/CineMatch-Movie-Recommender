import pandas as pd
import ast
import streamlit as st
import requests
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD ENV ----------------
load_dotenv("api.env")
API_KEY = os.getenv("TMDB_API_KEY")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="CineMatch", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
h1 { color: #E50914; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 CineMatch - Smart Movie Recommender")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    credits = pd.read_csv("credits.csv")

    movies = movies.merge(credits, left_on="id", right_on="movie_id")

    movies = movies[[
        "id",
        "title_x",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]]

    movies.columns = [
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]

    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)

    return movies

movies = load_data()

# ---------------- DATA PROCESSING ----------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i["name"].replace(" ", ""))
    return L

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(lambda x: convert(x)[:3])

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            return i["name"].replace(" ", "")
    return ""

movies["crew"] = movies["crew"].apply(fetch_director)

# Create tags properly
movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["crew"]
)

# ---------------- SIMILARITY ----------------
@st.cache_resource
def create_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(data["tags"]).toarray()
    return cosine_similarity(vectors)

similarity = create_similarity(movies)

# ---------------- FETCH POSTER ----------------
def fetch_poster(movie_id):
    if not API_KEY:
        return "https://via.placeholder.com/300x450?text=API+Key+Missing"

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return "https://via.placeholder.com/300x450?text=No+Image"

        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            return "https://via.placeholder.com/300x450?text=No+Image"

    except:
        return "https://via.placeholder.com/300x450?text=Error"

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie, num_recommendations):
    if movie not in movies["title"].values:
        return [], []

    index = movies[movies["title"] == movie].index[0]
    distances = list(enumerate(similarity[index]))

    movies_list = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:num_recommendations+1]

    names = []
    posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters

# ---------------- RECOMMEND BY GENRE ----------------
def recommend_by_genre(selected_genre):
    genre_movies = movies[movies["genres"].apply(lambda x: selected_genre in x)]

    names = []
    posters = []

    for _, row in genre_movies.head(10).iterrows():
        names.append(row["title"])
        posters.append(fetch_poster(row["movie_id"]))

    return names, posters

# ---------------- UI ----------------
tab1, tab2 = st.tabs(["🎥 Recommend by Movie", "🎭 Recommend by Genre"])

# ---------- TAB 1 ----------
with tab1:
    selected_movie = st.selectbox(
        "Choose a movie",
        sorted(movies["title"].unique())
    )

    num_recs = st.slider("Number of Recommendations", 5, 20, 10)

    if st.button("Recommend Similar Movies"):
        with st.spinner("Finding similar movies..."):
            names, posters = recommend(selected_movie, num_recs)

            if names:
                cols = st.columns(5)
                for i in range(len(names)):
                    col = cols[i % 5]
                    with col:
                        st.image(posters[i])
                        st.caption(names[i])
            else:
                st.warning("Movie not found!")

# ---------- TAB 2 ----------
with tab2:
    all_genres = sorted(set(g for sublist in movies["genres"] for g in sublist))

    selected_genre = st.selectbox("Select Genre", all_genres)

    names, posters = recommend_by_genre(selected_genre)

    cols = st.columns(5)
    for i in range(len(names)):
        col = cols[i % 5]
        with col:
            st.image(posters[i])
            st.caption(names[i])