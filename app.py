# app.py
from flask import Flask, request, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
import requests
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

TMDB_API_KEY = '8265bd1679663a7ea12ac168da84d2e8&language=en-US'  # Kendi TMDB API anahtarınızı ekleyin

with open('movies.pkl', 'rb') as f:
    movies = pickle.load(f)
with open('user_movie_matrix.pkl', 'rb') as f:
    user_movie_matrix = pickle.load(f)
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

movie_id_map = dict(zip(movies['title'], movies['movie_id']))


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


class UserWatchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Float, nullable=True)
    comment = db.Column(db.Text, nullable=True)


with app.app_context():
    db.create_all()


def fetch_poster(movie_id):
    if not movie_id:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    except requests.RequestException as e:
        print(f"Error fetching poster for movie_id {movie_id}: {e}")
        return None


def recommend(user_id, num_recommendations=10):
    user_history = UserWatchHistory.query.filter_by(user_id=user_id).all()
    if not user_history:
        return [{'title': "İzleme geçmişi bulunamadı.", 'movie_id': None, 'poster_url': None}]

    user_movie_matrix_temp = user_movie_matrix.copy()
    for watch in user_history:
        if watch.movie_id in user_movie_matrix_temp.columns:
            user_movie_matrix_temp.at[user_id, watch.movie_id] = watch.rating or 0

    user_movie_matrix_sparse = csr_matrix(user_movie_matrix_temp.values)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_movie_matrix_sparse)

    n_samples = user_movie_matrix_temp.shape[0]
    n_neighbors = min(5, n_samples)
    distances, indices = knn.kneighbors(user_movie_matrix_temp.loc[user_id].values.reshape(1, -1),
                                        n_neighbors=n_neighbors)

    similar_users = [user_movie_matrix_temp.index[i] for i in indices.flatten()]
    similar_user_movies = pd.DataFrame([
        {'user_id': watch.user_id, 'movie_id': watch.movie_id, 'rating': watch.rating}
        for watch in UserWatchHistory.query.filter(UserWatchHistory.user_id.in_(similar_users)).all()
    ])
    user_watched = [watch.movie_id for watch in user_history]

    if similar_user_movies.empty:
        # Yalnızca NLP tabanlı öneri yap
        last_watched_movie_id = user_history[-1].movie_id
        if last_watched_movie_id not in movies['movie_id'].values:
            return [{'title': f"Son izlenen film (movie_id: {last_watched_movie_id}) veri setinde bulunamadı.",
                     'movie_id': None, 'poster_url': None}]
        idx = movies[movies['movie_id'] == last_watched_movie_id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]

        recommendations = []
        for i, score in sim_scores:
            movie_id = movies.iloc[i]['movie_id']
            movie_title = movies.iloc[i]['title']
            poster_url = fetch_poster(movie_id)
            recommendations.append({'title': movie_title, 'movie_id': movie_id, 'poster_url': poster_url})
        return recommendations

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
        poster_url = fetch_poster(movie_id)
        recommendations.append({'title': movie_title, 'movie_id': movie_id, 'poster_url': poster_url})

    return recommendations[:num_recommendations]


@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')

        if User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first():
            return render_template('register.html', error="Kullanıcı adı veya e-posta zaten kayıtlı.")
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('recommend_page'))
        return render_template('login.html', error="Geçersiz kimlik bilgileri.")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/recommend', methods=['GET', 'POST'])
def recommend_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    recommendations = []
    selected_movie = None
    movies_list = movies['title'].tolist()
    user_id = session['user_id']

    if request.method == 'POST':
        movie = request.form['movie']
        selected_movie = movie
        movie_id = movie_id_map.get(movie)
        if movie_id:
            try:
                rating = request.form.get('rating', type=float)
                comment = request.form.get('comment', '').strip()
                existing_watch = UserWatchHistory.query.filter_by(user_id=user_id, movie_id=movie_id).first()
                if not existing_watch:
                    new_watch = UserWatchHistory(user_id=user_id, movie_id=movie_id, rating=rating, comment=comment)
                    db.session.add(new_watch)
                    db.session.commit()
                    recommendations = recommend(user_id)
                else:
                    return render_template('movie_recommender.html', recommendations=recommendations, movies=movies_list,
                                           selected_movie=selected_movie, message="Film zaten izleme geçmişinde.")
            except Exception as e:
                db.session.rollback()
                return render_template('movie_recommender.html', recommendations=recommendations, movies=movies_list,
                                       selected_movie=selected_movie, error=f"İzleme hatası: {str(e)}")
        else:
            return render_template('movie_recommender.html', recommendations=recommendations, movies=movies_list,
                                   selected_movie=selected_movie, error="Seçilen film bulunamadı.")
    else:
        recommendations = recommend(user_id)

    return render_template('movie_recommender.html', recommendations=recommendations, movies=movies_list,
                           selected_movie=selected_movie)


@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    watch_history = UserWatchHistory.query.filter_by(user_id=session['user_id']).all()
    history_data = []
    for watch in watch_history:
        movie_title = movies[movies['movie_id'] == watch.movie_id]['title'].iloc[0] if watch.movie_id in movies[
            'movie_id'].values else "Bilinmiyor"
        poster_url = fetch_poster(watch.movie_id)
        history_data.append({
            'title': movie_title,
            'rating': watch.rating,
            'comment': watch.comment,
            'id': watch.id,
            'poster_url': poster_url
        })
    return render_template('watch_history.html', history=history_data)


if __name__ == '__main__':
    app.run(debug=True)