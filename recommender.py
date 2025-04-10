import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# ------------------------
# Step 1: Load the dataset
# ------------------------
print("📦 Loading MovieLens 100K...")

# Ratings
ratings = pd.read_csv("data/u.data", sep="\t", names=['user_id', 'item_id', 'rating', 'timestamp'])

# Movie titles
movie_titles = pd.read_csv(
    "data/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1]
)
movie_titles.columns = ["item_id", "title"]

# Merge ratings and titles
data = pd.merge(ratings, movie_titles, on="item_id")

# -------------------------------
# Step 2: Train the SVD Recommender
# -------------------------------
print("🧠 Training SVD model...")

reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data[["user_id", "item_id", "rating"]], reader)

trainset, testset = train_test_split(surprise_data, test_size=0.2)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)
print("✅ RMSE:", accuracy.rmse(predictions))

# -------------------------------
# Step 3: Recommend Movies
# -------------------------------
def recommend_movies(user_id, n=10):
    all_movie_ids = movie_titles["item_id"].unique()
    watched = data[data["user_id"] == user_id]["item_id"].tolist()
    unseen = [mid for mid in all_movie_ids if mid not in watched]

    # Predict for all unseen movies
    predicted_ratings = [model.predict(user_id, mid) for mid in unseen]
    predicted_ratings.sort(key=lambda x: x.est, reverse=True)

    top_n_ids = [pred.iid for pred in predicted_ratings[:n]]
    return movie_titles[movie_titles["item_id"].isin(top_n_ids)]["title"].tolist()

# -------------------------------
# Step 4: Try it Out
# -------------------------------
user_input = input("Enter a user ID (1–943): ")

try:
    uid = int(user_input)
    print(f"\n🎬 Top 10 Recommendations for User {uid}:\n")
    for i, movie in enumerate(recommend_movies(uid), 1):
        print(f"{i}. {movie}")
except ValueError:
    print("⚠️ Please enter a valid numeric user ID (1–943).")
