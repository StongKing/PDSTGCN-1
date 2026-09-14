"""Compatibility helper. data_generate.py already writes the Divvy static matrices."""
import configparser, pickle
from lib.utils import get_adjacency_matrix

c=configparser.ConfigParser(); c.read('configurations/DIVVY_astgcn.conf')
d=c['Data']; A,D=get_adjacency_matrix(d['adj_filename'], int(d['num_of_vertices']), d.get('id_filename', fallback=None))
with open('data/DIVVY/adj_DIVVY.pkl','wb') as f: pickle.dump(A,f)
with open('data/DIVVY/adj_DIVVY_distance.pkl','wb') as f: pickle.dump(D,f)
print('saved static adjacency pickles:', A.shape)
