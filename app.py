import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# -----------------------------
# Load and prepare MovieLens data
# -----------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("data/u.data", sep="\t", names=['user_id', 'item_id', 'rating', 'timestamp'])
    movie_titles = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1])
    movie_titles.columns = ["item_id", "title"]
    data = pd.merge(ratings, movie_titles, on="item_id")
    return data, movie_titles

# -----------------------------
# Train the model using SVD
# -----------------------------
@st.cache_resource
def train_model(data):
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(data[["user_id", "item_id", "rating"]], reader)
    trainset, _ = train_test_split(surprise_data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    return model

# -----------------------------
# Generate Recommendations
# -----------------------------
def recommend_movies(user_id, model, data, movie_titles, n=10):
    all_movie_ids = movie_titles["item_id"].unique()
    watched = data[data["user_id"] == user_id]["item_id"].tolist()
    unseen = [mid for mid in all_movie_ids if mid not in watched]

    predictions = [model.predict(user_id, mid) for mid in unseen]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n_ids = [pred.iid for pred in predictions[:n]]
    return movie_titles[movie_titles["item_id"].isin(top_n_ids)]["title"].tolist()

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🎥 Movie Recommender System")
st.write("Get personalized movie recommendations using collaborative filtering!")

data, movie_titles = load_data()
model = train_model(data)

user_id = st.number_input("Enter a User ID (1–943)", min_value=1, max_value=943, step=1)

if st.button("Recommend Movies"):
    with st.spinner("Generating recommendations..."):
        recommendations = recommend_movies(user_id, model, data, movie_titles)
        st.success(f"Top 10 Movie Recommendations for User {user_id}")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
