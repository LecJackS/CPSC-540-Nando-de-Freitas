from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from .tree import InternalNode
from .tree import LeafNode

class TreeBuilder(object):

    def _find_split_parameters(self, X, Y, n_min_leaf, n_trials):
        """
        Compute parameters of the best split for the data X, Y.

        X: features, one data point per row
        Y: labels, one data point per row
        n_trials: the number of split dimensions to try.
        n_min_leaf: the minimum leaf size -- don't create a split with
            children smaller than this.

        Returns the pair (split_dim, split_threshold) or None if no appropriate
        split is found.  split_dim is an integer and split_threshold is a real
        number.

        Call self._information_gain(Y, Y_left, Y_right) to compute the
        information gain of a split.
        """
        """
        y
        ^
        |x x  | o  x    
        |    x|     o  
        |x    |o   o   
        |_____|_______>
              |     x
        """
        # Instead of n_trials, I search for ALL points and dimensions
        X_len = X.shape[0]
        X_dim = X.shape[1]
        max_score = -np.float('inf')
        FOUND = False
        for d in range(X_dim):
            for i in range(X_len):
                # each X[i][d] is an split candidate
                Y_left  = Y[X[:,d] <  X[i,d], :]
                Y_right = Y[X[:,d] >= X[i,d], :]
                if (len(Y_left) <= n_min_leaf) or (len(Y_right) <= n_min_leaf):
                    # This split is no good, continue with next one
                    continue
                score = self._information_gain(Y, Y_left, Y_right)
                if score > max_score:
                    FOUND = True
                    split_dim = d
                    split_threshold = X[i, d]
                    max_score = score
                    #print('New split')
        if FOUND:
            #print("X: ", X.shape)
            #print("y: ", Y.shape)            
            #print('best split dim/threshold: ', split_dim, split_threshold)
            #plt.scatter(X[:, split_dim], np.argmax(Y[:], axis=1))
            #plt.scatter(split_threshold, 0, c='r')
            #plt.show()
            return (split_dim, split_threshold)
        else:
            return None


    def fit(self, X, Y, max_depth, n_min_leaf, n_trials):
        yhat = Y.mean(axis=0).reshape((1,-1))

        # short circuit for pure leafs
        if np.all(Y == Y[0]):
            return LeafNode(yhat)

        # avoid growing trees that are too deep
        if max_depth <= 0:
            return LeafNode(yhat)

        split_params = self._find_split_parameters(
                X, Y, n_min_leaf=n_min_leaf, n_trials=n_trials)

        # if we didn't find a good split point then become leaf
        if split_params is None:
            return LeafNode(yhat)

        split_dim, split_threshold = split_params

        mask_l = X[:,split_dim] < split_threshold
        mask_r = np.logical_not(mask_l)

        # refuse to make leafs that are too small
        if np.sum(mask_l) < n_min_leaf or \
                np.sum(mask_r) < n_min_leaf:
            raise Exception("Leaf too small")

        # otherwise split this node recursively
        left_child = self.fit(
                X[mask_l],
                Y[mask_l],
                max_depth=max_depth - 1,
                n_min_leaf=n_min_leaf,
                n_trials=n_trials)

        right_child = self.fit(
                X[mask_r],
                Y[mask_r],
                max_depth=max_depth - 1,
                n_min_leaf=n_min_leaf,
                n_trials=n_trials)

        return InternalNode(
                dim=split_dim,
                threshold=split_threshold,
                left_child=left_child,
                right_child=right_child)


class ClassificationTreeBuilder(TreeBuilder):

    def _entropy(self, x):
        x = x[x>0]
        return -np.sum(x*np.log(x))

    def _information_gain(self, y, y_l, y_r):
        n = y.shape[0]
        n_l = y_l.shape[0]
        n_r = y_r.shape[0]

        H = self._entropy(y.mean(axis=0))
        H_l = self._entropy(y_l.mean(axis=0))
        H_r = self._entropy(y_r.mean(axis=0))

        return H - n_l/n * H_l - n_r/n * H_r


class RegressionTreeBuilder(TreeBuilder):

    def _information_gain(self, y, y_l, y_r):
        assert y.size == y_l.size + y_r.size
        assert y_l.size > 0
        assert y_r.size > 0

        sse = np.sum((y - y.mean())**2)
        sse_l = np.sum((y_l - y_l.mean())**2)
        sse_r = np.sum((y_r - y_r.mean())**2)

        return sse - sse_l - sse_r

