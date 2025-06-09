from flask import Flask, request, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Güvenli bir anahtar kullanın
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Pickle dosyalarını yükleme
with open('new_df.pkl', 'rb') as f:
    new_df = pickle.load(f)
with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

# Kullanıcı modeli
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Kullanıcı izleme geçmişi modeli
class UserWatchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Float, nullable=True)

# Veritabanını oluştur
with app.app_context():
    db.create_all()

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

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        
        # Kullanıcı zaten var mı kontrol et
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
    
    if request.method == 'POST':
        movie = request.form['movie']
        selected_movie = movie
        recommendations = recommend(movie)
        
        # Kullanıcı izleme geçmişine ekleme
        movie_id = new_df[new_df['title'] == movie]['movie_id'].iloc[0] if movie in new_df['title'].values else None
        if movie_id:
            new_watch = UserWatchHistory(user_id=session['user_id'], movie_id=movie_id)
            db.session.add(new_watch)
            db.session.commit()
    
    movies = new_df['title'].tolist()  # Film listesi dropdown için
    return render_template('recommend.html', recommendations=recommendations, movies=movies, selected_movie=selected_movie)

if __name__ == '__main__':
    app.run(debug=True)