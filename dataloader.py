import pickle
import os

data_path = "data/"

def load_pickle(child_path):
    # Direct load dari bucket, file storage
    try:
        data = pickle.load(open(os.path.join(data_path, child_path), 'rb'))
        return data
    except Exception as e:
        print(e)