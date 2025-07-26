import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load your data
ratings = pd.read_csv("D:\\Datasets\\ratings\\ratings.csv")
movies = pd.read_csv("D:\\Datasets\\Movies\\movies.csv")

# Filter out invalid movie IDs
valid_movie_ids = set(movies['movieId'])
ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]

# Create mappings and matrix
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

# Movie title dictionary
movie_titles = dict(zip(movies['movieId'], movies['title']))

# KNN model
def find_similar_movies(movie_id, X, k=5, metric='cosine'):
    neighbour_ids = []
    if movie_id not in movie_mapper:
        return []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in neighbour[0]:
        neighbour_ids.append(movie_inv_mapper[i])
    neighbour_ids.pop(0)
    return neighbour_ids

# Streamlit UI
st.title("🎬 Movie Recommender System")

# Select user
user_ids = sorted(ratings['userId'].unique())
user_id = st.selectbox("Select a User ID", user_ids)

if st.button("Recommend"):
    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
        st.warning("No ratings found for selected user.")
    else:
        top_rated = user_ratings[user_ratings['rating'] == user_ratings['rating'].max()]
        movie_id = top_rated['movieId'].iloc[0]
        movie_name = movie_titles.get(movie_id, f"Movie ID {movie_id}")

        st.write(f"Since you liked **{movie_name}**, you might also like:")
        similar_ids = find_similar_movies(movie_id, X, k=5)
        for mid in similar_ids:
            st.markdown(f"- {movie_titles.get(mid, f'Movie ID {mid}')}")

