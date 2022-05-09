from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


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
        # calculating means + probability:
        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))
        self.pi_ = np.zeros(len(self.classes_))
        for k in self.classes_:
            for i in range(X.shape[1]):
                c_avg = 0
                count = 0
                for j, s in enumerate(X):
                    if y[j] == k:
                        count += 1
                        c_avg += s[i]
                self.mu_[k][i] = c_avg / count
                self.pi_[k] = count / len(y)

        # calculating variance:
        self.vars_ = np.zeros((len(self.classes_), X.shape[1]))
        for k in self.classes_:
            X_k = X[y == k]
            for feature in range(X_k.shape[1]):
                self.vars_[k][feature] = np.var(X_k[:, feature], ddof=1)

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
        likelihood_matrix = self.likelihood(X)
        return np.array([np.argmax(self.pi_ * likelihood_matrix[i]
                                   / (self.pi_ @ likelihood_matrix[i]))
                         for i, s in enumerate(X)])

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
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        likelihood_matrix = np.zeros((X.shape[0], len(self.classes_)))
        for i in self.classes_:
            d = X - self.mu_[i]
            likelihood_matrix[:, i] = np.exp(
                -0.5 * np.diag(d @ inv(np.diag(self.vars_[i])) @ d.T))
            likelihood_matrix[:, i] /= np.sqrt(
                (2 * np.pi) ** X.shape[1] * det(np.diag(self.vars_[i])))
        return likelihood_matrix

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
        from ...metrics import misclassification_error
        return misclassification_error(self.predict(X), y)
