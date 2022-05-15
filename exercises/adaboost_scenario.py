import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = \
        generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    iterations = range(1, n_learners + 1)
    train_err = [adaboost.partial_loss(train_X, train_y, t) for t in
                 iterations]
    test_err = [adaboost.partial_loss(test_X, test_y, t) for t in
                iterations]

    fig = go.Figure(
        [go.Scatter(x=list(iterations), y=train_err, name="train"),
         go.Scatter(x=list(iterations), y=test_err, name="test")],
        layout=go.Layout(
            title=r"$\text{Training and test errors as a function of the number of fitted learners}$",
            xaxis=dict(title=r"$\text{Number of fitted learners}$"),
            yaxis=dict(title=r"$\text{Error rate}$")))
    fig.show()
    fig.write_image(f"./ex4Plots/Training and test errors by learners.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"ensemble size = {t}"
                                        for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda X: adaboost.partial_predict(X, t),
                              lims[0], lims[1],
                              showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                        showlegend=False,
                        marker=dict(color=test_y,
                                    colorscale=[custom[0],
                                                custom[-1]],
                                    line=dict(color="black",
                                              width=1)))
             ],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries Of different ensemble sizes}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()
    fig.write_image(
        f"./ex4Plots/Decision Boundaries Of different ensemble sizes.png")

    # Question 3: Decision surface of best performing ensemble
    min = float("inf")
    optimal_size = 0
    for i, err in enumerate(test_err):
        if err < min:
            min = err
            optimal_size = i + 1

    fig = go.Figure(
        [decision_surface(lambda X: adaboost.partial_predict(X, optimal_size),
                          lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=test_y,
                                colorscale=[custom[0],
                                            custom[-1]],
                                line=dict(color="black",
                                          width=1)))
         ],
        layout=go.Layout(
            title=rf"$\textbf{{Ensemble of size {optimal_size} achieved the accuracy of {1 - min}}}$"))

    fig.show()
    fig.write_image(
        f"./ex4Plots/Best ensemble size and error-wise.png")

    # Question 4: Decision surface with weighted samples
    sizes = adaboost.D_ / np.max(adaboost.D_) * 20
    fig = go.Figure(
        [decision_surface(adaboost.predict,
                          lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=train_y, size=sizes,
                                colorscale=[custom[0],
                                            custom[-1]],
                                line=dict(color="black",
                                          width=1)))
         ],
        layout=go.Layout(
            title=rf"$\textbf{{Training set with a point size proportional to it’s weight in the last iteration}}$"))

    fig.show()
    fig.write_image(
        f"./ex4Plots/final training with sizes.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    # fit_and_evaluate_adaboost(noise=0.4)
