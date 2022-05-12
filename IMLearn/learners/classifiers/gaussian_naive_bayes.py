from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics import misclassification_error

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.mu_ = np.array(self.mu_)
        
        self.vars_ = []
        for k,i in enumerate(self.classes_):
            vars_j = []
            for j in range(X.shape[1]):
                vars_j.append(np.sum((X[y==i][:,j]-self.mu_[k,:][j]) ** 2)/(len(X[y==i][:,j])-1))
            self.vars_.append(vars_j)
        self.vars_ = np.array(self.vars_)
        
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
        for k,i in enumerate(self.classes_):
            LL_k = np.ones((X.shape[0]))
            for j in range(X.shape[1]):
                LL_k = LL_k*np.exp(- (X[:,j] - self.mu_[k,j]) ** 2 / (2 * self.vars_[k,j])) / np.sqrt(2 * np.pi * self.vars_[k,j])
            LL.append(LL_k)
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
