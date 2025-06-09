import pandas as pd
import numpy as np
import ast
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Verileri yükleme
credits = pd.read_csv('data/tmdb_5000_credits.csv')
movies = pd.read_csv('data/tmdb_5000_movies.csv')

# Verileri birleştirme
movies = movies.merge(credits, on='title')

# Gerekli sütunları seçme
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Eksik verileri temizleme
movies.dropna(inplace=True)

# JSON string'lerini listeye dönüştürme
def convert(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Cast: İlk 3 oyuncuyu alma
def convert_cast(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            l.append(i['name'])
            counter += 1
    return l

movies['cast'] = movies['cast'].apply(convert_cast)

# Crew: Sadece yönetmeni alma
def fetch_director(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l

movies['crew'] = movies['crew'].apply(fetch_director)

# Overview'ı kelimelere ayırma
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Boşlukları kaldırma (örneğin, "Sam Worthington" -> "SamWorthington")
def remove_space(word):
    l = []
    for i in word:
        l.append(i.replace(" ", ""))
    return l

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

# Tags oluşturma
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Yeni DataFrame
new_df = movies[['movie_id', 'title', 'tags']].copy()

# Tags'i string'e dönüştürme
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Küçük harfe çevirme
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Stemming
ps = PorterStemmer()

def stems(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

new_df['tags'] = new_df['tags'].apply(stems)

# Vektörleştirme
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

# Cosine similarity
similarity = cosine_similarity(vector)

# Vektör ve benzerlik matrisini kaydetme (tekrar hesaplamamak için)
with open('vector.pkl', 'wb') as f:
    pickle.dump(vector, f)
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)
with open('new_df.pkl', 'wb') as f:
    pickle.dump(new_df, f)

# Öneri fonksiyonu
def recommend(movie, num_recommendations=5):
    try:
        index = new_df[new_df['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommendations = []
        for i in distances[1:num_recommendations+1]:
            recommendations.append(new_df.iloc[i[0]].title)
        return recommendations
    except IndexError:
        return ["Film bulunamadı."]