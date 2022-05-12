import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        
        self.D_ = np.zeros((self.iterations_,y.shape[0]))
        D = np.ones(y.shape[0])/y.shape[0]
        for t in range(self.iterations_):
#             print(t)
            self.D_[t] = D
#             print(f'D {t}: {D}')
            self.models_.append(self.wl_().fit(X,D*y))
#             print(f'y_hat {t}: {self.models_[t].predict(X)}')
            e_t = np.sum((self.models_[t].predict(X)!=y)*1 * D)
#             print(f'epsilon {t}: {e_t}')
            self.weights_[t] = 0.5 * np.log((1.0-e_t) / e_t)
#             print(f'weight {t}: {self.weights_[t]}')
            D = D * np.exp((-1) * self.weights_[t] * y * self.models_[t].predict(X))
            D = D / np.sum(D)
        
#         raise NotImplementedError()

    def _predict(self, X):
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
        
        return self.partial_predict(X,self.iterations_)
        
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

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        
        h_final = np.zeros(X.shape[0])
        for t in range(T):
            h_t = self.models_[t].predict(X)*self.weights_[t]
            h_final = h_final+h_t
        h_final = np.sign(h_final)
        
        return h_final
        
#         raise NotImplementedError()

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        
        return misclassification_error(self.partial_predict(X,T),y,True)
        
#         raise NotImplementedError()
