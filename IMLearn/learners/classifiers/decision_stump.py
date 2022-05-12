from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        
        potenital_stumps = []
        for i in range(X.shape[1]):
            for s in [-1,1]:
                thr, thr_err = self._find_threshold(X[:,i],y,s)
                potenital_stumps.append(np.array([i,s,thr, thr_err]))
        potenital_stumps = np.array(potenital_stumps)

        lowest_err = np.argmin(potenital_stumps, axis=0)[3]
        self.j_, self.sign_, self.threshold_,lowest_err = potenital_stumps[lowest_err]
        self.j_ = int(self.j_)
        
#         raise NotImplementedError()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        
        decision_on_data_ZeroOrOne = (X[:, self.j_] >= self.threshold_)
        decision_on_data_OneOrMinusOne = (decision_on_data_ZeroOrOne*2)-1
        return self.sign_*decision_on_data_OneOrMinusOne
        
#         raise NotImplementedError()

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """                             
                                        
        #sorting labels by values
        values, labels = values[np.argsort(values[:])], labels[np.argsort(values[:])]
        #finding index to put threshold on
        maximum_cumsum = np.argmax(np.cumsum(labels * sign))
        #calculating threshold
        if maximum_cumsum==0:
            self.threshold_ = -np.inf 
        elif maximum_cumsum==values.shape[0]-1:
            self.threshold_ = np.inf    
        else: 
            self.threshold_ = (values[maximum_cumsum]+values[maximum_cumsum])/2
        #caculating loss
        loss = np.sum([labels == sign]) - np.cumsum(labels * sign)[maximum_cumsum]

        
        return self.threshold_, loss
#         raise NotImplementedError()


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        
        return misclassification_error(self._predict(X),y,True)
        
#         raise NotImplementedError()
