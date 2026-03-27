from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

app = Flask(__name__)

df = pd.read_csv("data/music_clusters.csv")

model = joblib.load('models/kmeans_model.pkl')

numerical_features = [
    "valence",
    "danceability",
    "energy",
    "tempo",
    "acousticness",
    "liveness",
    "speechiness",
    "instrumentalness",
]

scaler = StandardScaler()
df_scaled_values = scaler.fit_transform(df[numerical_features])
df_scaled = pd.DataFrame(df_scaled_values, columns=numerical_features, index=df.index)

all_titles = sorted(df['name'].unique().tolist())

def recommend_songs(song_name, df, df_scaled, num_recommendations=5):
    song_name_clean = song_name.strip().lower()
    name_mask = df['name'].str.strip().str.lower() == song_name_clean
    match_indices = df.index[name_mask].tolist()
    
    if not match_indices:
        raise ValueError(f"Song '{song_name}' not found in the database. Please check the Library section for available titles!")

    source_idx = match_indices[0]
    song_cluster = df.at[source_idx, "Cluster"]
    
    cluster_indices = df.index[df["Cluster"] == song_cluster].tolist()
    cluster_features_scaled = df_scaled.loc[cluster_indices]
    similarity = cosine_similarity(cluster_features_scaled, cluster_features_scaled)
    
    pos_in_matrix = cluster_indices.index(source_idx)
    similar_pos = np.argsort(similarity[pos_in_matrix])[-(num_recommendations + 1) : -1][::-1]
    
    final_df_indices = [cluster_indices[p] for p in similar_pos]
    
    results = df.loc[final_df_indices, ["name", "year", "artists"]].copy()
    results['artists'] = results['artists'].apply(lambda x: x.strip("[]").replace("'", "").replace('"', ''))
    
    return results.to_dict('records')

@app.route("/")
def index():
    return render_template("index.html", all_songs=all_titles)

@app.route("/recommend", methods=["POST"])
def recommend():
    recommendations = []
    error_message = None
    song_name = request.form.get("song_name")
    try:
        recommendations = recommend_songs(song_name, df, df_scaled)
    except Exception as e:
        error_message = str(e)
            
    return render_template("index.html", 
                           recommendations=recommendations, 
                           error_message=error_message, 
                           all_songs=all_titles,
                           searched_song=song_name)

@app.route("/suggest")
def suggest():
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify([])
    
    suggestions = [t for t in all_titles if query in t.lower()]
    return jsonify(suggestions[:10])

if __name__ == "__main__":
    app.run(debug=True)
