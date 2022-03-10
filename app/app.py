import json
import os
import pathlib
import pickle

import pandas as pd
import numpy as np
import sys
import traceback

# contains 5 important columns - 'title', 'vote_count', 'vote_average', 'year', 'id', 'movieId'.
smd = pd.read_csv(os.path.join(pathlib.Path(__file__).parent.absolute(), 'smd.csv'), header=0,
                  index_col=0,
                  squeeze=True)

# Pre computed cosine similarity Numpy arrays.
cosine_sim = np.load(os.path.join(pathlib.Path(__file__).parent.absolute(), 'cosine_sim.npy'))


def load(file_name):
    dump_obj = pickle.load(open(file_name, 'rb'))
    return dump_obj['algo']


# https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD
# It takes in the user id and the movie id.
svd = load(os.path.join(pathlib.Path(__file__).parent.absolute(), 'svd'))


def lambda_handler(event, context):
    top_10_movies = ""
    try:
        print(f"event={event}")
        title = event['multiValueQueryStringParameters']['Title'][0]
        userId = event['multiValueQueryStringParameters']['UserId'][0]
        print(f'Received Userid={userId}, Title={title}')
        found_row = smd[smd['title'] == title]
        idx = 0
        cosine_idx = 0
        for key, value in found_row.iterrows():
            idx = value['id']
            cosine_idx = key
            break
        print("STAGE 0 complete - Found the corresponding idx={}, cosine_idx={} for the title {}."
              .format(idx, cosine_idx, title))

        # STAGE 1: Filter out top 50 movies based on cosine similarity.
        sim_scores = list(enumerate(cosine_sim[int(cosine_idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:50]
        # print(f'sim_scores for the ={sim_scores}')

        movie_indices = [i[0] for i in sim_scores]
        print(f"Stage 1 complete - Found movie indices = {movie_indices} for top 50 titles that share the same cosine "
              f"similarity with the passed in movie index.")

        # STAGE 2: Use the SVD model that is built on 9220 movies
        # For each of the 20 top movies that share cosine similarity with the passed in movie title,
        # Fetch the title, vote count, vote average, year and id.
        movies = smd.loc[movie_indices]
        print("Stage 2 completed. Successfully fetched the title, vote count, vote average, year and id for the "
              "top 50 movies. ")
        # print(movies)
        # STAGE 3: Apply SVD algorithm predict() API call to find the estimation for the top 20 movie titles.
        # movies['id'].apply(lambda x: print(f'x={x}'))
        # print(movies[['movieId']])
        movies['est'] = movies['movieId'].apply(lambda x: svd.predict(int(userId), x).est)

        movies = movies.sort_values(['est', 'year'], ascending=False)
        # print('movies={}'.format(movies[['title', 'est']]))
        print("Stage 3 completed. Successfully applied SVD predict() on the list of 50 movie indices.")

        # STAGE 4: Pick top 10 movies from this list and send this back to the customer.
        top_10_movies = movies.head(10)['title'].tolist()
        print("Stage 4 completed. Successfully fetched top 10 movies from the list and returning this back to the "
              "customer.")
        print('TOP 10 movie recommendations={}'.format(top_10_movies))

    except Exception:
        # printing stack trace
        traceback.print_exception(*sys.exc_info())
        print(traceback.format_exc())
    finally:
        return {
            'statusCode': 200,
            'body': json.dumps(
                {
                    "top_10_movies": top_10_movies,
                }
            )
        }
