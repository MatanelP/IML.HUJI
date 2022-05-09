import numpy as np
import sklearn.discriminant_analysis

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


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
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    fig = go.Figure(
        layout=go.Layout(title="perceptron loss"
                               " values as a function of the"
                               " training iterations",
                         xaxis_title="training iterations",
                         yaxis_title="loss values"))
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        Perceptron(
            callback=lambda fit, x, t: losses.append(fit.loss(X, y))).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig.add_trace(
            go.Scatter(x=list(range(1, len(losses) + 1)), y=losses,
                       mode='lines',
                       name=n))
    fig.write_image(f"./ex3Plots/run_perceptron.png")
    fig.show()


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
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.2, .2])
        # Fit models and predict over training set

        classifiers = [GaussianNaiveBayes().fit(X, y), LDA().fit(X, y)]
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        accuracies = [round(accuracy(y, classifiers[i].predict(X)), 3) for i in
                      range(len(classifiers))]
        model_names = [f"Gaussian Naive Bayes, accuracy: {accuracies[0]}",
                       f"LDA, accuracy: {accuracies[1]}"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=model_names,
                            horizontal_spacing=0.01)
        fig.update_layout(title=f"Classifying over dataset: {f}",
                          margin=dict(t=100))  # , height= 1000, width = 500)

        # Add traces for data-points setting symbols and colors
        for i, classifier in enumerate(classifiers):
            means = classifier.mu_
            fig.add_traces(
                [decision_surface(classifier.predict, lims[0], lims[1],
                                  showscale=False),
                 go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                            showlegend=False,
                            marker=dict(color=y,
                                        symbol=class_symbols[y],
                                        colorscale=class_colors(3),
                                        line=dict(color="black",
                                                  width=1))),
                 # Add `X` dots specifying fitted Gaussians' means
                 go.Scatter(x=means[:, 0], y=means[:, 1], mode="markers",
                            name="mean",
                            marker=dict(color="black", symbol="x", size=8)),
                 # Add ellipses depicting the covariances of the fitted Gaussians
                 ] +
                [get_ellipse(mean, classifier.cov_ if i
                else np.diag(classifier.vars_[j]))
                 for j, mean in enumerate(means)],
                rows=1, cols=1 + i),

            fig.update_layout(width=1500, height=750).update_xaxes(
                visible=False).update_yaxes(visible=False)
        fig.write_image(f"./ex3Plots/LDA+GNB for {f}.png")
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    GaussianNaiveBayes().fit(np.array([1,1,1,2,2,3,2,4,3,3,3,4]).reshape(-1,2), np.array([0,0,1,1,1,1]))
    run_perceptron()
    compare_gaussian_classifiers()
