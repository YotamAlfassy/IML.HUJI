from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
<<<<<<< Updated upstream
=======
from IMLearn.metrics import accuracy
import numpy as np
>>>>>>> Stashed changes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
<<<<<<< Updated upstream
from math import atan2, pi
=======
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt
import math
>>>>>>> Stashed changes


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
<<<<<<< Updated upstream
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)
=======
    arr = np.load(filename)
    X = arr[:,:2]
    y = arr[:,2] 
    
    return [X,y]
#     raise NotImplementedError()
>>>>>>> Stashed changes


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset          
        X1,y1 = load_dataset(f'datasets/{f}')
#         raise NotImplementedError()

        # Fit Perceptron and record loss in each fit iteration
        loss_array = []
        def callback_loss(p: Perceptron,x: np.ndarray ,y: int):
            loss_array.append(p.loss(X1,y1))

        P = Perceptron(callback=callback_loss)
        P.fit(X1,y1)
#         raise NotImplementedError()

<<<<<<< Updated upstream
        # Plot figure of loss as function of fitting iteration
        raise NotImplementedError()
=======
        # Plot figure
        loss_array = np.array(loss_array)
        x = np.linspace(1,len(loss_array),len(loss_array))
        
        plt.figure(figsize=(20,8))
        plt.plot(x, loss_array, 'o', color='black')
        plt.ylim([0, 1])
        plt.title(f'{n} loss by iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
#         raise NotImplementedError()
>>>>>>> Stashed changes


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X,y = load_dataset(f'datasets/{f}')
#         raise NotImplementedError()

        # Fit models and predict over training set
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        x = np.linspace(1,X.shape[0],X.shape[0])
        fig = plt.figure(figsize=(20,8))
        AX = [fig.add_subplot(122),fig.add_subplot(121)]
        
        for i in range(2):
            model = LDA() if i == 0 else GaussianNaiveBayes()
            model.fit(X,y)
            name = "LDA" if i==0 else "GNB"
            
            AX[i].title.set_text(f'{name} Plot, accuracy: {accuracy(y,model.predict(X))}, dataset: {f}')

            y_2 = X[np.where(model.predict(X)==2)]
            y_1 = X[np.where(model.predict(X)==1)]
            y_0 = X[np.where(model.predict(X)==0)]
            AX[i].plot(y_1[:, :1], y_1[:, 1:2], 'co')
            AX[i].plot(y_0[:, :1], y_0[:, 1:2], "yo")
            AX[i].plot(y_2[:, :1], y_2[:, 1:2], "bo")
            AX[i].plot(X[:, :1], X[:, 1:2], 'r+')
            
            if i==0: 
                for j in range(3): _plot_gaussian(AX[i],model.mu_.T[j],model.cov_,"k")
            else: 
                for j in range(3): _plot_gaussian(AX[i],model.mu_[j],model.vars_[j],"k")     
            
        plt.show()
        #         raise NotImplementedError()


def _plot_gaussian(plot,mean, covariance, color, zorder=0):
    """Plots the mean and 2-std ellipse of a given Gaussian"""
    plot.plot(mean[0], mean[1], color[0] + "X", zorder=zorder)

    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    radius = np.sqrt(5.991)
    eigvals, eigvecs = np.linalg.eig(covariance)
    axis = np.sqrt(eigvals) * radius
    slope = eigvecs[1][0] / eigvecs[1][1]
    angle = 180.0 * np.arctan(slope) / np.pi

    
    u=mean[0]       #x-position of the center
    v=mean[1]     #y-position of the center
    a=axis[0]       #radius on the x-axis
    b=axis[1]   #radius on the y-axis
    t_rot=math.radians(angle)  #rotation angle

    t = np.linspace(0, 2*np.pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
         #u,v removed to keep the same center location
    R_rot = np.array([[np.cos(t_rot) , -np.sin(t_rot)],[np.sin(t_rot) , np.cos(t_rot)]])  
         #2-D rotation matrix

    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

    plot.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],'black' )

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    
#     gnb = GaussianNaiveBayes()
#     #Q1
#     S = np.array([(0,0),(1,0),(2,1),(3,1),(4,1),(5,1),(6,2),(7,2)])
#     X = np.array([S[:,0]]).T
#     y = S[:,1]
#     print(f'The estimated class probability of class 0: {round(gnb.fit(X,y).pi_[0],2)}')
#     print(f'The estimated class expectation of class 1: {round(gnb.fit(X,y).mu_[1][0],2)}')
    
#     #Q2
#     S = np.array([([1,1],0),([1,2],0),([2,3],1),([2,4],1),([3,3],1),([3,4],1)])
#     y = S[:,1]
#     X = []
#     for i in S[:,0]:
#         X.append(np.array(i))
#     X = np.array(X)
#     print(f'The estimated variance of feature 1 in class 0: {round(gnb.fit(X,y).vars_[0,1],2)}')
#     print(f'The estimated variance of feature 1 in class 1: {round(gnb.fit(X,y).vars_[1,1],2)}')   