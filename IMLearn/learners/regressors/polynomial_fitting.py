from __future__ import annotations
from typing import NoReturn
from . import LinearRegression
from ...base import BaseEstimator
from IMLearn.metrics import mean_square_error
import numpy as np


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self.coefs_, self.k_ = None,k
#         raise NotImplementedError()

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        
        X_transform = []       

        for j in range( self.k_ + 1 ) :
            x_pow = np.power( X, j )
            X_transform.append(x_pow)

        return np.transpose(np.array(X_transform))
        
#         raise NotImplementedError()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
#         X_transform = []       

#         for j in range( self.k_ + 1 ) :
#             x_pow = np.power( X, j )
#             X_transform.append(x_pow)

#         X_transform = np.transpose(np.array(X_transform))
        LR = LinearRegression(False)
        self.coefs_ = LR.fit(self.__transform(X),y).coefs_
#         self.coefs_ = LR.fit(X_transform,y).coefs_
        
#         raise NotImplementedError()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        
        return self.__transform(X)@(self.coefs_)
#         raise NotImplementedError()

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(self.predict(X),y)
#         raise NotImplementedError()
