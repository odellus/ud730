from six.moves import cPickle as pickle
import numpy as np

def load_pickle(fname='notMNIST.pickle'):
    """
    Function: load_pickle(fname):

    Description: load pickle file into memory

    Parameters: fname -- name of pickle file to load

    Returns: notMNIST -- a dictionary containing notMNIST data and labels
    """
    with open(fname,'r') as f:
        notMNIST = pickle.load(f)
    return notMNIST

def find_duplicates(dataset, labels):
    """
    Function: find_duplicates(dataset, labels)

    Description: remove duplicate entries from the notMNIST dataset

    Parameters: dataset -- the actual images in a numerical array
                labels  -- the corresponding labels of the images in dataset

    Return: dataset, lablels -- dataset and labels with duplicates removed
    """
    # create a dictionary(hash table)
    h = {}
    # store the locations of the duplicates for deletion
    dups = []
    # gather the shape of the dataset
    shape = dataset.shape
    # flatten tensor array down to a matrix if needed
    if len(shape) > 2:
        dataset = dataset.reshape((shape[0], shape[1]*shape[2]))
    # iterate through the dataset
    for k in range(shape[0]):
        # create a string that represents the image
        u = dataset[k,:].tostring()
        # check if the string is already a key in the dictionary
        if u not in h:
            # you've seen this now. make it a key in the dictionary
            h[u] = 1
        else:
            # now you've seen it again b/c it's a duplicate
            h[u] += 1
            # add this index as a duplicate
            dups.append(k)
    # delete the rows whose indices are contained in dups
    dataset = np.delete(dataset, dups, 0)
    # delete the elements of labels whose indices are in dups
    labels = np.delete(labels, dups, None)
    # pass back a cleaned dataset and corresponding labels
    return dataset, labels
