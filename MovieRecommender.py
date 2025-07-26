import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load ratings data
ratings = pd.read_csv("D:\\Datasets\\ratings\\ratings.csv")
print("Ratings DataFrame:")
print(ratings.head())

# Load movies data
movies = pd.read_csv("D:\\Datasets\\Movies\\movies.csv")
print("\nMovies DataFrame:")
print(movies.head())

# Basic stats from the ratings data
n_ratings = len(ratings)
n_movies = len(ratings['movieId'].unique())
n_users = len(ratings['userId'].unique())

print(f"\nNumber of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")

user_freq = ratings[['userId', 'movieId']].groupby(
    'userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
print(user_freq.head())

#ratings.groupby('movieId')[['rating']].mean(): groups the ratings by movieId and calculates the mean rating for each movie.
#mean_rating['rating'].idxmin(): finds the movie ID with the lowest average rating.
#mean_rating['rating'].idxmax(): finds the movie ID with the highest average rating.
#movies.loc[movies['movieId'] == lowest_rated]: filters the movies DataFrame to display information about the movie with the lowest rating.
#movies.loc[movies['movieId'] == highest_rated]: filters the movies DataFrame to display information about the movie with the highest rating.

mean_rating = ratings.groupby('movieId')[['rating']].mean()
lowest_rated = mean_rating['rating'].idxmin()
movies.loc[movies['movieId'] == lowest_rated]
highest_rated = mean_rating['rating'].idxmax()
movies.loc[movies['movieId'] == highest_rated]
ratings[ratings['movieId']==highest_rated]
ratings[ratings['movieId']==lowest_rated]

movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()

# User-Item Matrix Creation
from scipy.sparse import csr_matrix


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

# Movie Similarity Analysis
from sklearn.neighbors import NearestNeighbors


def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []

    if movie_id not in movie_mapper:
        print(f"Movie ID {movie_id} not found in movie_mapper!")
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

def recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10):
    df1 = ratings[ratings['userId'] == user_id]

    movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]

    movie_titles = dict(zip(movies['movieId'], movies['title']))

    similar_ids = find_similar_movies(movie_id, X, k)

    print(f"Since you watched {movie_titles[movie_id]}, you might also like:")

    for i in similar_ids:
        if i in movie_titles:
            print(movie_titles[i])

recommend_movies_for_user(877, X, user_mapper, movie_mapper, movie_inv_mapper, k=5)
