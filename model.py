import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, data_path='movies.csv'):
        self.df = pd.read_csv(data_path)
        self.df.dropna(inplace=True)
        self.cv = CountVectorizer(tokenizer=lambda x: x.split('|'))
        self.count_matrix = self.cv.fit_transform(self.df['genres'])
        self.similarity = cosine_similarity(self.count_matrix)

    def recommend(self, movie_title, top_n=3):
        if movie_title not in self.df['title'].values:
            return []

        index = self.df[self.df['title'] == movie_title].index[0]
        scores = list(enumerate(self.similarity[index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in scores[1:top_n + 1]]
        return self.df.iloc[top_indices]['title'].tolist()
