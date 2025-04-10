# 🎬 Movie Recommender System

A collaborative filtering-based movie recommendation engine built using [Surprise](https://surpriselib.com/) and deployed with Streamlit.

This project uses the [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) dataset and suggests top 10 movies for any user ID based on past ratings from other users.

---

## 💡 Features

- 🔎 Personalized movie recommendations
- ⚡ Fast training using SVD from the Surprise library
- 🖥️ Streamlit web app for instant interaction
- 🧪 CLI version for quick terminal use

---

## 🚀 Getting Started

### 1. Clone the repo & navigate into it:
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender

### 2. Set up a virtual environment:
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate        # On macOS/Linux
# OR
venv\Scripts\activate           # On Windows

### 3. Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt

### 4. Download the MovieLens 100K data:
Get it from: https://grouplens.org/datasets/movielens/100k/


