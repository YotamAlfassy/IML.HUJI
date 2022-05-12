from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        
        self.classes_ = np.unique(y)
        
        self.mu_ = []
        for i in self.classes_:
            self.mu_.append(X[y==i].mean(axis=0))
        self.mu_ = np.array(self.mu_).T
        
        self.cov_ = np.zeros((X.shape[1],X.shape[1]))
        for j,i in enumerate(self.classes_):
            a = (X[y==i]-self.mu_[:,j])
            self.cov_ += a.T@(a/(y.size-self.classes_.size))
        self.cov_= np.array(self.cov_)
        
        self._cov_inv = np.linalg.inv(self.cov_)
        
        self.pi_ = []
        for i in self.classes_:
            self.pi_.append((y==i).mean())
        self.pi_ = np.array(self.pi_)
        
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
        
        LL = self.likelihood(X)
        for i in range(LL.shape[1]):
            LL[:,i]= LL[:,i]*self.pi_[i]
        for i in range(LL.shape[0]):
            LL[i,:]= LL[i,:]/np.sum(LL,axis=1)[i]
        return np.take(self.classes_, np.argmax(LL,axis=1))
        
#         raise NotImplementedError()

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
            
            
        LL = []
        for i,x in enumerate(self.classes_):
            mahalanobis = np.einsum("bi,ij,bj->b", X-self.mu_[:,i], self._cov_inv, X-self.mu_[:,i])
            LL_i =  np.exp(-.5 * mahalanobis) / np.sqrt((2*np.pi) ** len(X) * np.linalg.det(self.cov_))
            LL.append(LL_i)
        return np.array(LL).T

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
