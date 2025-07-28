from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

# Load Data
ratings = pd.read_csv("D:\\Datasets\\ratings\\ratings.csv")
movies = pd.read_csv("D:\\Datasets\\Movies\\movies.csv")

# Preprocessing
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []
    if movie_id not in movie_mapper:
        return []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)

    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_id = int(request.form["user_id"])
        k = int(request.form["k"])

        if user_id not in ratings['userId'].values:
            return render_template("index.html", error="User ID not found!")

        df1 = ratings[ratings['userId'] == user_id]
        if df1.empty:
            return render_template("index.html", error="No ratings found for this user!")

        movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
        similar_ids = find_similar_movies(movie_id, X, k)

        movie_titles = dict(zip(movies['movieId'], movies['title']))
        recommendations = [movie_titles[i] for i in similar_ids if i in movie_titles]
        watched_movie = movie_titles[movie_id]

        return render_template("index.html", watched_movie=watched_movie, recommendations=recommendations)

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
