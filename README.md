# MovieRecommender
A content-based movie recommendation system built with Python, Pandas, and scikit-learn. Uses K-Nearest Neighbors and a sparse user-item matrix to suggest similar movies based on user preferences.
📊 Detailed Analysis
This project implements a content-based movie recommender system using collaborative filtering and K-Nearest Neighbors (KNN) on the MovieLens dataset. The goal is to recommend movies to a user based on their highest-rated movie by identifying similar movies through cosine similarity on a sparse user-item rating matrix.

📁 Dataset Description
The system uses two CSV files:

ratings.csv: Contains user-movie rating interactions with fields: userId, movieId, rating, timestamp.

movies.csv: Contains metadata about each movie: movieId, title, and genres.

After preprocessing:

Ratings from movies not present in the movies.csv file are excluded.

The resulting dataset includes only valid user-movie interactions.

📈 Exploratory Data Insights
Total number of ratings: X

Total unique users: Y

Total unique movies: Z

Average ratings per user: A

Average ratings per movie: B

These statistics provide an overview of the sparsity of the dataset and help in understanding user behavior.

🧮 Matrix Creation
A sparse user-item matrix is created using scipy.sparse.csr_matrix, where:

Rows represent movies.

Columns represent users.

The values in the matrix represent the rating a user gave to a movie.

This sparse representation is memory-efficient and suitable for large-scale similarity calculations.

🤖 Recommendation Engine
The core recommendation engine consists of:

KNN (K-Nearest Neighbors) from scikit-learn: Calculates cosine similarity between movies.

Similarity Function: Given a movie ID, the system retrieves the top K most similar movies (excluding itself).

User-Based Recommendation: For a given user, the system identifies their highest-rated movie and recommends K similar movies.

✅ Error Handling and Robustness
Handles KeyError when a movie is not found in the metadata.

Filters out invalid movie IDs during preprocessing to maintain consistency.

Uses .get() for safe dictionary access when fetching movie titles.

📌 Key Functions
create_matrix(df): Builds the sparse user-item matrix and mapping dictionaries.

find_similar_movies(movie_id, X, k): Returns the top K similar movies using cosine similarity.

recommend_movies_for_user(user_id, X, ...): Given a user, recommends similar movies based on their preferences.
