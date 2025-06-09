# add_test_data.py
from app import app, db, User, UserWatchHistory
from werkzeug.security import generate_password_hash
import pickle

# movies.pkl dosyasını yükle
with open('movies.pkl', 'rb') as f:
    movies = pickle.load(f)

# Gerçek movie_id'lerden bir örnek al
valid_movie_ids = movies['movie_id'].head(7).tolist()
print("Kullanılan movie_id'ler:", valid_movie_ids)

with app.app_context():
    # Önce mevcut verileri sil
    db.drop_all()
    db.create_all()

    # Test kullanıcıları ekle
    user1 = User(username="user1", email="user1@example.com", password=generate_password_hash("password1"))
    user2 = User(username="user2", email="user2@example.com", password=generate_password_hash("password2"))
    user3 = User(username="user3", email="user3@example.com", password=generate_password_hash("password3"))
    user4 = User(username="user4", email="user4@example.com", password=generate_password_hash("password4"))
    user5 = User(username="user5", email="user5@example.com", password=generate_password_hash("password5"))
    db.session.add_all([user1, user2, user3, user4, user5])
    db.session.commit()

    # Test izleme geçmişi ekle (gerçek movie_id'lerle)
    watch1 = UserWatchHistory(user_id=user1.id, movie_id=valid_movie_ids[0], rating=4.0, comment="Harika bir film!")
    watch2 = UserWatchHistory(user_id=user1.id, movie_id=valid_movie_ids[1], rating=3.5, comment="İyiydi.")
    watch3 = UserWatchHistory(user_id=user2.id, movie_id=valid_movie_ids[0], rating=5.0, comment="Mükemmel!")
    watch4 = UserWatchHistory(user_id=user2.id, movie_id=valid_movie_ids[2], rating=4.5, comment="Güzeldi.")
    watch5 = UserWatchHistory(user_id=user3.id, movie_id=valid_movie_ids[1], rating=3.0, comment="Fena değil.")
    watch6 = UserWatchHistory(user_id=user4.id, movie_id=valid_movie_ids[2], rating=4.0, comment="Sevdim!")
    watch7 = UserWatchHistory(user_id=user5.id, movie_id=valid_movie_ids[0], rating=4.5, comment="Çok iyi!")
    db.session.add_all([watch1, watch2, watch3, watch4, watch5, watch6, watch7])
    db.session.commit()

print("Test verileri eklendi!")