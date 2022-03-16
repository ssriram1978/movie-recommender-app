import json
import os
import pathlib
import pickle

import pandas as pd
import numpy as np
import sys
import traceback

movie_db = pd.read_csv(os.path.join(pathlib.Path(__file__).parent.absolute(), 'ml-25m.csv'), header=0,
                       index_col=0,
                       squeeze=True)

# Pre-computed cosine similarity Numpy arrays.
loaded = np.load(os.path.join(pathlib.Path(__file__).parent.absolute(), 'cosine_sim_25m.npz'))
cosine_sim = loaded['arr_0']


def load(file_name):
    dump_obj = pickle.load(open(os.path.join(pathlib.Path(__file__).parent.absolute(), file_name), 'rb'))
    return dump_obj['algo']


svd = load('svd')


def lambda_handler(event, context):
    top_10_movies = ""
    try:
        print(f"event={event}")
        title = event['multiValueQueryStringParameters']['Title'][0]
        userId = event['multiValueQueryStringParameters']['UserId'][0]
        print(f'Received Userid={userId}, Title={title}')
        idx = movie_db.index[movie_db['title'] == title]

        if not len(idx.values):
            # STAGE 1:
            # Check if there is a keyword match for the starting name of the movie.
            print('Unable to find exact title match')
            print('Trying to search for the starting keyword in the title name.')
            movies = movie_db[movie_db['title'].apply(lambda y: str(y).lower().startswith(str(title).lower()))]

            # STAGE 2:
            # Check if there is a keyword match in the tags for the movies.
            if movies.shape[0] < 20:
                print('Trying to search for keywords in the tags column.')
                tag_match = movie_db[movie_db['combination'].apply(lambda y: str(title) in str(y))]
                movies = pd.concat([movies, tag_match], ignore_index=True, axis=0)
            print(movies[['title', 'rating']])
            print("Stage 2 completed. Successfully fetched the title, vote count, vote average, year and id for the "
                  "top matched movies. ")
        else:
            idx = idx.values[0]
            print("STAGE 0 complete - Found the corresponding idx={} for the title {}."
                  .format(idx, title))

            # STAGE 1: Filter out top 20 movies based on cosine similarity.
            sim_scores = list(enumerate(cosine_sim[int(idx)]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:20]
            movie_indices = [i[0] for i in sim_scores]
            print(
                f"Stage 1 complete - Found movie indices = {movie_indices} for top 50 titles that share the same cosine "
                f"similarity with the passed in movie index.")

            # STAGE 2:
            # For each of the 20 top movies that share cosine similarity with the passed in movie title,
            # Fetch the title, vote count, vote average, year and id.
            movies = movie_db.loc[movie_indices]
            print(movies[['title', 'rating']])
            print("Stage 2 completed. Successfully fetched the title, vote count, vote average, year and id for the "
                  "top 20 movies. ")

        # STAGE 3: Use the SVD model that is built 62,000 movies by 162,000 users.
        # Apply SVD algorithm predict() API call to find the estimation for the top 50 movie titles.
        movies['est'] = movies['movieId'].apply(lambda x: svd.predict(int(userId), x).est)
        movies = movies.sort_values(['est', 'year'], ascending=False)
        # print('movies={}'.format(movies[['title', 'est']]))
        print(movies[['title', 'est', 'rating']])
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