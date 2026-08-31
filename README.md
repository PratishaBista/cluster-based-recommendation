# Music Recommendation System (K-Means & Cosine Similarity)

A content-based music recommendation system that clusters tracks by acoustic features and retrieves nearest neighbors using cosine similarity. Built with Python, scikit-learn, and Flask.

The system uses unsupervised machine learning to group tracks based on high-dimensional audio profiles. When a user searches for a track, the system identifies its cluster and calculates intra-cluster cosine similarity across standardized feature vectors to generate ranked recommendations.

## Tech Stack

- Language: Python (3.9+)
- Machine Learning: scikit-learn (K-Means, StandardScaler, Cosine Similarity), joblib
- Data Processing: pandas, NumPy
- Web Framework: Flask, Jinja2
- Frontend: HTML5, CSS3, JavaScript (Fetch API for search autocomplete)
- Data Source and APIs: Spotify Audio Features Dataset, Spotipy

## System Architecture and Algorithm

1. Feature Standardization:
   Eight acoustic dimensions (`valence`, `danceability`, `energy`, `tempo`, `acousticness`, `liveness`, `speechiness`, `instrumentalness`) are normalized using `StandardScaler` to ensure zero mean and unit variance.

2. Clustering (K-Means):
   The dataset is partitioned into acoustic clusters using K-Means clustering (`models/kmeans_model.pkl`).

3. Retrieval and Ranking:
   - The query track is resolved to its assigned cluster.
   - Feature vectors for all tracks in that cluster are extracted from the scaled feature matrix.
   - Cosine similarity is computed between the target track and cluster candidates.
   - The top N nearest tracks are ranked and returned.

```
[Query: Song Title]
       │
       ▼
[Resolve Track & Cluster ID]
       │
       ▼
[Filter Scaled Matrix by Cluster]
       │
       ▼
[Compute Pairwise Cosine Similarity]
       │
       ▼
[Rank & Return Top N Nearest Tracks]
```

## Project Structure

```
├── app.py                  # Flask server and recommendation routing logic
├── requirements.txt        # Project dependencies
├── data/
│   ├── music_clusters.csv  # Preprocessed dataset with cluster labels
│   └── songs_dataset.csv   # Raw audio features dataset
├── models/
│   └── kmeans_model.pkl    # Serialized scikit-learn K-Means model
├── notebooks/
│   └── kmeans_clustering_analysis.ipynb  # EDA, elbow method, and model training
├── static/                 # CSS stylesheets, client scripts, and assets
└── templates/
    └── index.html          # Web interface with autocomplete search
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/PratishaBista/kmeans-recommender.git
cd kmeans-recommender
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the web application:
```bash
python app.py
```
The server runs locally at `http://127.0.0.1:5000/`.
****
## Endpoints

- `GET /`: Renders the search interface and song catalog.
- `POST /recommend`: Accepts `song_name` and returns the top 5 recommended tracks.
- `GET /suggest?q=<query>`: Returns a JSON array of matching song titles for live autocomplete.
