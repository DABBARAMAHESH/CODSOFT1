import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Example movie dataset
data = {
    'title': ['The Matrix', 'John Wick', 'Inception', 'Interstellar', 'The Notebook'],
    'description': [
        'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
        'An ex-hitman comes out of retirement to track down the gangsters that killed his dog.',
        'A thief who steals corporate secrets through dream-sharing technology is given a chance to erase his criminal history.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity’s survival.',
        'A young couple fall in love in the 1940s but are separated by social differences and war.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Vectorize the descriptions using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Function to get recommendations
def recommend(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df.index[df['title'] == title][0]

    # Get similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 3 most similar movies (excluding itself)
    sim_scores = sim_scores[1:4]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 3 most similar movies
    return df['title'].iloc[movie_indices]


# Example usage
print("Recommended movies for 'The Matrix':")
print(recommend('The Matrix'))
