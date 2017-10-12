import numpy as np
from past.builtins import xrange
from scipy import stats


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      test_datapoint = X[i]
      for j in xrange(num_train):
        # Compute the l2 distance between the ith test point and the jth
        # training point, and store the result in dists[i, j].
        dists[i, j] = np.linalg.norm(test_datapoint - self.X_train[j], ord=2)
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """

    # X = test points. there are 500, each has 3072 dims.
    # self.X_train has 5000, each with 3072 dims.
    # goal is to return the an array of 500x5000 - distances between each test points and all training points in X_train.
    # so broadcast the thing?

    # X[i] is my test point: 3072, 1
    # self.X_train are my training examples: 5000, 3072
    # goal is to get result that is 1, 5000

    # np.expand_dims(X[i], 1) => (3072, 1)
    # self.X_train.T - np.expand_dims(X[i], 1) => (3072, 5000)

    # vector_diffs = (self.X_train.T - np.expand_dims(X[i], 1)) => (5000, 3072)
    # np.linalg.norm(vector_diffs.T, axis=1, ord=2)

    # returns

    # dists is where i store distances: will be 500, 5000

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################

      # OK, data in X_train has some dimensionality, but is of the same length as y_train.
      vector_diffs = (self.X_train.T - np.expand_dims(X[i], 1))
      dists[i] = np.linalg.norm(vector_diffs.T, axis=1, ord=2)
    # print(dists.shape)
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """

    # X is 500, 3072
    # X_train is 5000, 3072

    # don't want to just broadcast. kind of want to concat

    # goal: get a 500x5000x3072 matrix, where the last is 3072-3072.
    # then squash it to 500x5000x1 via norm, then squeeze dims.

    # could also broadcast the 3072s
    # sqrt(all elems squared and added)


    # dists = np.zeros((num_test, num_train))

    X = np.expand_dims(X, axis=1)
    X_train = np.expand_dims(self.X_train, axis=0)
    return np.linalg.norm(X - X_train, axis=2)

    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    # return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """

    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)

    for i in xrange(num_test):
      sorted_indexes = np.argsort(dists[i])[:k]
      closest_y = self.y_train[sorted_indexes]
      mode, _ = stats.mode(closest_y)
      y_pred[i] = mode[0]

    return y_pred

