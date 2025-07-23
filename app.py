from flask import Flask, request, jsonify
from model import Recommender

app = Flask(__name__)
rec_engine = Recommender()

@app.route('/')
def home():
    return '🎬 Movie Recommendation API is running!'

@app.route('/recommend', methods=['GET'])
def recommend():
    movie = request.args.get('movie')
    if not movie:
        return jsonify({'error': 'Please provide a movie title'}), 400

    recommendations = rec_engine.recommend(movie)
    return jsonify({
        'input_movie': movie,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
