# movie_recommendation.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Veri setlerini yükle
credits = pd.read_csv('data/tmdb_5000_credits.csv')
movies = pd.read_csv('data/tmdb_5000_movies.csv')

# 'id' sütununu 'movie_id' olarak yeniden adlandır
movies = movies.rename(columns={'id': 'movie_id'})

# Birleştirme
movies = movies.merge(credits, on='title')

# 'movie_id_x' sütununu 'movie_id' olarak yeniden adlandır ve gereksiz sütunları kaldır
movies = movies.rename(columns={'movie_id_x': 'movie_id'})
movies = movies.drop(columns=['movie_id_y'], errors='ignore')

# Gerekli sütunları seç
movies = movies[['movie_id', 'title', 'overview']]
movies.dropna(subset=['overview'], inplace=True)

# Gerçek movie_id'lerden bir örnek al (ilk 12 film)
valid_movie_ids = movies['movie_id'].head(12).tolist()
print("Kullanılan movie_id'ler:", valid_movie_ids)

# Simüle edilmiş kullanıcı izleme geçmişi (gerçek movie_id'lerle)
user_watch_history = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    'movie_id': valid_movie_ids,  # Gerçek movie_id'ler
    'rating': [4.0, 3.5, 5.0, 4.5, 3.0, 4.0, 3.5, 4.0, 5.0, 3.0, 4.5, 3.5]
})

# Kullanıcı-film matrisini oluştur
user_movie_matrix = user_watch_history.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
user_movie_matrix_sparse = csr_matrix(user_movie_matrix.values)

# NLP için overview vektörleştirme
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Collaborative filtering model
n_samples = user_movie_matrix_sparse.shape[0]
n_neighbors = min(5, n_samples)
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_movie_matrix_sparse)


# Öneri fonksiyonu
def recommend(user_id, num_recommendations=10):
    if user_id not in user_movie_matrix.index:
        return [{'title': "Kullanıcı izleme geçmişi bulunamadı.", 'movie_id': None}]

    distances, indices = knn.kneighbors(user_movie_matrix.loc[user_id].values.reshape(1, -1), n_neighbors=n_neighbors)
    similar_users = [user_movie_matrix.index[i] for i in indices.flatten()]

    similar_user_movies = user_watch_history[user_watch_history['user_id'].isin(similar_users)]
    user_watched = user_watch_history[user_watch_history['user_id'] == user_id]['movie_id'].tolist()

    if similar_user_movies.empty:
        return [{'title': "Benzer kullanıcıların izleme geçmişi bulunamadı.", 'movie_id': None}]

    recommended_movie_ids = similar_user_movies[~similar_user_movies['movie_id'].isin(user_watched)][
        'movie_id'].value_counts().head(num_recommendations).index

    recommendations = []
    for movie_id in recommended_movie_ids:
        # movie_id'nin movies DataFrame'inde olup olmadığını kontrol et
        if movie_id not in movies['movie_id'].values:
            print(f"movie_id {movie_id} movies DataFrame'inde bulunamadı.")
            continue
        idx = movies[movies['movie_id'] == movie_id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        movie_title = movies.iloc[idx]['title']
        recommendations.append({'title': movie_title, 'movie_id': movie_id})

    return recommendations[:num_recommendations]


# Verileri kaydet
with open('movies.pkl', 'wb') as f:
    pickle.dump(movies, f)
with open('user_movie_matrix.pkl', 'wb') as f:
    pickle.dump(user_movie_matrix, f)
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
with open('cosine_sim.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)