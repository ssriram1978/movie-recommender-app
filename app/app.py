import json

import pandas as pd
from surprise import dump
import numpy as np

smd = pd.read_csv('https://s3.amazonaws.com/movielens.data/smd.csv', header=0, parse_dates=[0], index_col=0,
                  squeeze=True)
indices = pd.read_csv('https://s3.amazonaws.com/movielens.data/indices.csv', header=0, parse_dates=[0], index_col=0,
                      squeeze=True)
id_map = pd.read_csv('https://s3.amazonaws.com/movielens.data/id_map.csv', header=0, parse_dates=[0], index_col=0,
                     squeeze=True)
# https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD
_, svd = dump.load('svd')
cosine_sim = np.load('cosine_sim.npy')


def lambda_handler(event, context):
    top_10_movies = ""
    try:
        print(f"event={event}")
        title = event['multiValueQueryStringParameters']['Title'][0]
        userId = event['multiValueQueryStringParameters']['UserId'][0]
        idx = indices[title]
        print(f'Received Userid={userId}, Title={title}')
        indices_map = id_map.set_index('id')

        # STAGE 1: Filter out top 26 movies based on cosine similarity.
        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        # STAGE 2: Use the SVD model that is built on 9220 movies,
        movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
        movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)
        top_10_movies = movies.head(10)['title'].tolist()
    except:
        pass
    finally:
        return {
            'statusCode': 200,
            'body': json.dumps(
                {
                    "top_10_movies": top_10_movies,
                }
            )
        }
