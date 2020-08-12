""" 
make a pickled file of [
    (t1_path, t2_path, seg_path),
    ...
]


"""
import os
import pickle
import numpy as np

repo_dir = '/home/kyle/MedicalZooPytorch'
repo_database = os.path.join(repo_dir, 'datasets')
database = '/mnt/data/Database'

# from database to repo_database
directory_map = {
    'iseg_2017': 'iseg_2017',
    'iSeg2019': 'iseg_2019',
    'mrbrains18': 'mrbrains_2018'
}
    
for database_dir, repo_database_dir in directory_map.items():
    
    database_dir = os.path.join(database, database_dir)
    repo_database_dir = os.path.join(repo_database, repo_database_dir)

    for folder in os.listdir(database_dir):
        # filename is iSeg-2017-Testing, iSeg-2017-Training, iSeg-2017-Validation

        folder_path = os.path.join(database_dir, folder)

        data = []
        for filename in os.listdir(folder_path):
            if '.img' in filename:
                with open(filename, 'rb') as f:
                    data = np.fromfile(f, np.dtype('>u2'))
                    print(f, data.shape)
                    # T1 and T2 are shape 7077888,
                    # label is shape 3538944,
                    image = data.reshape((144,192,256))


        # store npy files here
        training_files = os.path.join(iseg17_training, 'npy_files')
        output_path = os.path.join(iseg17_training, 'iseg2017-list-train-samples-1024.txt')

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

