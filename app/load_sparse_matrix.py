import os
import pathlib

from sklearn.metrics.pairwise import cosine_similarity
import scipy
from psutil import virtual_memory
import numpy as np

import pandas as pd


def check_ram():
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

    if ram_gb < 20:
        print('Not using a high-RAM runtime')
    else:
        print('You are using a high-RAM runtime!')


class OnlineRecommendations:
    def __init__(self):
        self._count_matrix = None
        self._md2 = None
        self._cosine_sim = None

    def prepare_cosine_similarity(self):
        self._count_matrix = scipy.sparse.load_npz('count_matrix.npz')
        self._cosine_sim = cosine_similarity(self._count_matrix, self._count_matrix)
        np.savez_compressed('cosine_sim_25m.npz', self._cosine_sim)

    def load_cosine_similarity(self):
        loaded = np.load('cosine_sim_25m.npz')
        self._cosine_sim = loaded['arr_0']

    def get_recommendations(self, title):
        if not self._md2:
            self._md2 = pd.read_csv(os.path.join(pathlib.Path(__file__).parent.absolute(), 'md2.csv'),
                                    header=0,
                                    index_col=0)
        if not self._cosine_sim:
            self.load_cosine_similarity()

        idx = self._md2.index[self._md2['title'] == title]
        print(f'idx = {idx}')
        print(f'type(idx)={type(idx)}, idx={idx[0]}')
        sim_scores = list(enumerate(self._cosine_sim[idx[0]]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        print(f'movie_indices={movie_indices}')
        return self._md2.iloc[movie_indices]


if __name__ == '__main__':
    recommendations = OnlineRecommendations()
    print(recommendations.get_recommendations('Father of the Bride Part II').head(10))
