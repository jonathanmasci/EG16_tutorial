import time
import logging
import numpy as np
import h5py
import theano
import os
import scipy.sparse as sp


def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_file, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
            #out = sp.csc_matrix((data, ir, jc), shape=(80*6890, 6890)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


class ClassificationDatasetPatchesMinimal(object):
    def __init__(self, train_txt, test_txt, descs_path, patches_path, geods_path, labels_path,
                 desc_field='desc', patch_field='M', geod_field='geods', label_field='labels',
                 epoch_size=100):
        self.train_txt = train_txt
        self.test_txt = test_txt
        self.descs_path = descs_path
        self.patches_path = patches_path
        self.geods_path = geods_path
        self.labels_path = labels_path
        self.desc_field = desc_field
        self.patch_field = patch_field
        self.geod_field = geod_field
        self.label_field = label_field
        self.epoch_size = epoch_size

        self.train_data = []
        self.test_data = []
        self.train_labels = []
        self.test_labels = []
        self.train_patches = []
        self.test_patches = []

        with open(self.train_txt, 'r') as f:
            self.train_fnames = [line.rstrip() for line in f]
        with open(self.test_txt, 'r') as f:
            self.test_fnames = [line.rstrip() for line in f]

        print("Loading train descs")
        tic = time.time()
        for name in self.train_fnames:
            self.train_data.append(load_matlab_file(os.path.join(self.descs_path, name), self.desc_field).squeeze())
        print "elapsed time %f" % (time.time() - tic)
        print("Loading test descs")
        tic = time.time()
        for name in self.test_fnames:
            self.test_data.append(load_matlab_file(os.path.join(self.descs_path, name), self.desc_field).squeeze())
        print "elapsed time %f" % (time.time() - tic)

        print "Loading train patches"
        tic = time.time()
        for name in self.train_fnames:
            self.train_patches.append(load_matlab_file(os.path.join(self.patches_path, name), self.patch_field))
        print "elapsed time %f" % (time.time() - tic)
        print "Loading test patches"
        tic = time.time()
        for name in self.test_fnames:
            self.test_patches.append(load_matlab_file(os.path.join(self.patches_path, name), self.patch_field))
        print "elapsed time %f" % (time.time() - tic)

        print "Loading train labels"
        tic = time.time()
        for name in self.train_fnames:
            self.train_labels.append(load_matlab_file(os.path.join(self.labels_path, name),
                                                      self.label_field).astype(np.int32).flatten() - 1)
        print "elapsed time %f" % (time.time() - tic)
        print "Loading test labels"
        tic = time.time()
        for name in self.test_fnames:
            self.test_labels.append(load_matlab_file(os.path.join(self.labels_path, name),
                                                     self.label_field).astype(np.int32).flatten() - 1)
        print "elapsed time %f" % (time.time() - tic)

    def get_data_ndims(self):
        return self.train_data[0].shape[1]

    def train_iter(self):
        for nd in xrange(self.epoch_size):
            i = np.random.permutation(len(self.train_data))[0]
            yield (self.train_data[i], self.train_patches[i], self.train_labels[i])

    def test_iter(self):
        for i in xrange(len(self.test_data)):
            yield (self.test_data[i], self.test_patches[i], self.test_labels[i])
