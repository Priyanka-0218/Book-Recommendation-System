import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import ast
import pickle
import sqlite3
import os

class UserProfiling:
    def __init__(self, location_of_df='/home/gm/Desktop/ExcelR_Projects/book_recommendation/model_training/Model_building_Hybrid/TF-IDF/bart_final_preprocess.csv'):
        self.df = pd.read_csv(location_of_df)

    def save_pickle(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def get_item_profile(self):
        # Ensure the use of the class attribute self.df
        df = self.df.copy()
        
        df = df[~df.book_id.duplicated()]
        genres = df['genres'].apply(ast.literal_eval)

        # Convert genres to binary feature vectors
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(genres)
        genre_features = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=df['book_id'])

        # Reduce dimensions with PCA
        pca = PCA(n_components=100)  # Adjust the number of components as needed
        reduced_genre_features = pca.fit_transform(genre_features)

        # Convert back to DataFrame for easy handling
        reduced_genre_features = pd.DataFrame(reduced_genre_features, index=df['book_id'])

        return reduced_genre_features

    def get_reviewer_profile(self):
        # import sqlite3
        # import ast
        connection=sqlite3.connect('/home/gm/Desktop/ExcelR_Projects/book_recommendation/preprocessing_cleaning/FINAL_DATA/DATASETS/book_reviews.db')
        book_reviews=pd.read_sql_query("SELECT * FROM book_reviews", connection)

        book_reviews=book_reviews[~book_reviews['review_rating'].isna()]
        book_reviews.reset_index(inplace=True,drop=True)
        book_reviews['rating_of_user']=book_reviews['review_rating'].apply(lambda x: x.split()[1])
        # user_item_interaction=book_reviews.groupby('reviewer_id')[['book_id', 'rating_of_user']].apply(lambda x: x.reset_index(drop=True))
        a=book_reviews[['book_id', 'reviewer_id', 'rating_of_user']]
        df_evaluated = a.map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # print(df_evaluated)
        return df_evaluated

    def filtered_data(self):
        item_profiles = self.get_item_profile()
        df_evaluated = self.get_reviewer_profile()

        total_index = set(item_profiles.index) ^ set(df_evaluated.book_id)
        symmetric_difference_list = list(total_index)

        item_profiles_filtered = item_profiles[~item_profiles.index.isin(symmetric_difference_list)]
        df_evaluated_filtered = df_evaluated[~df_evaluated['book_id'].isin(symmetric_difference_list)]

        return df_evaluated_filtered, item_profiles_filtered

    def get_user_profiles(self):
        df_evaluated_filtered, item_profiles_filtered = self.filtered_data()
        user_item_interactions = {
            reviewer_id: dict(zip(group['book_id'], group['rating_of_user']))
            for reviewer_id, group in df_evaluated_filtered.groupby('reviewer_id')
        }

        user_profiles = {}
        for user, interactions in user_item_interactions.items():
            profile = np.zeros(item_profiles_filtered.shape[1])
            total_weight = 0

            for book_id, rating in interactions.items():
                if book_id in item_profiles_filtered.index:
                    profile += rating * item_profiles_filtered.loc[book_id]
                    total_weight += rating

            if total_weight > 0:
                user_profiles[user] = profile / total_weight
            else:
                user_profiles[user] = profile

        user_profiles = pd.DataFrame(user_profiles).T

        return user_profiles, item_profiles_filtered

    def create_profiles(self):
        user_profiles, item_profiles_filtered = self.get_user_profiles()
        # Save both profiles in a single pickle file
        self.save_pickle({'user_profiles': user_profiles, 'item_profiles_filtered': item_profiles_filtered}, 'profiles.pkl')

    def load_profiles(self):
        profiles = self.load_pickle('profiles.pkl')
        return profiles['user_profiles'], profiles['item_profiles_filtered']

    def recommend_books(self, user_id=None, book_id=None, top_n=10):
        user_profiles, item_profiles = self.load_profiles()

        if user_id is not None:
            if user_id not in user_profiles.index:
                return [], []
            user_profile = user_profiles.loc[user_id].values.reshape(1, -1)
            similarities = cosine_similarity(user_profile, item_profiles)
        elif book_id is not None:
            if book_id not in item_profiles.index:
                return [], []
            item_profile = item_profiles.loc[book_id].values.reshape(1, -1)
            similarities = cosine_similarity(item_profile, item_profiles)
        else:
            return [], []

        similar_items = np.argsort(similarities[0])[::-1][:top_n]
        recommended_book_ids = item_profiles.index[similar_items]
        similarity_scores = similarities[0][similar_items]
        
        return recommended_book_ids, similarity_scores

