from fastapi import FastAPI, Depends
app = FastAPI()
import pickle
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import implicit
import pandas as pd

movies = pd.read_csv("Data\items.csv")
ratings = pd.read_csv(r"Data\ratings.csv")

@app.get('/collaborative/{user_id}')
async def root1(user_id):
    user_item = sparse.load_npz("Data\collaborative.npz")
    model = pickle.load(open('Models\model_collaborative.sav', 'rb'))
    recommended, _ =  zip(*model.recommend(int(user_id)-1, user_item))
    recommend_frame = []
    for val in recommended:
        movie_idx = ratings.iloc[val]['movie_id']
        idx = movies[movies['movie_id'] == movie_idx].index
        recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0]})
    df = pd.DataFrame(recommend_frame,index=range(1,11))
    return {'Top 10 Prediction for user id: {}'.format(user_id): df} 

@app.get('/content/{input}')
async def root2(input):
    movie = pickle.load(open('Data\content.pkl', 'rb'))
    model = pickle.load(open('Models\content_based.sav', 'rb'))
    _, indices = model.kneighbors(movie,n_neighbors=11)
    movie_ = input
    input = movies[movies['title'].str.contains(input)]
    input_id = input.iloc[0]['movie_id']
    indices = indices[input_id-1]
    recommend_frame = []
    for val in indices[1:]:
        idx = movies.iloc[val]['movie_id']
        recommend_frame.append({'Title':movies.iloc[idx]['title']})
    df = pd.DataFrame(recommend_frame,index=range(1,11))
    return {'Top 10 Prediction for movie: {}'.format(movie_): df}
    