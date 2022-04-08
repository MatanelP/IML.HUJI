from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)
    # remove duplicates and empty entries
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    # remove features that should value positively but are not
    for feature in ["price", "sqft_living", "sqft_lot",
                    "floors", "yr_built", "zipcode"]:
        data.drop(data.loc[data[feature] <= 0].index, inplace=True)
    # adding is_new feature based on built year and renovation year
    data["is_new"] = np.where(data["yr_built"] > 2010, 1, 0)
    data["is_new"] = np.where(data["yr_renovated"] > 2010, 1, 0)
    # zipcode encoding
    data = pd.get_dummies(data, prefix="zipcode", columns=["zipcode"])
    # extracting price as response vector
    prices = data["price"]
    # dropping id, date and price
    data.drop(["id", "date", "price"], axis=1, inplace=True)
    return data, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = np.std(y)
    for feat in X:
        feat_std = np.std(X[feat])
        corr = np.cov(X[feat], y)[0][1] / (feat_std * y_std)
        go.Figure(data=go.Scatter(x=X[feat], y=y, mode="markers")) \
            .update_layout(title=f"Pearson Correlation of prices and {feat} is"
                                 f" {round(corr, 4)}", xaxis={"title": feat},
                           yaxis={"title": "price"}) \
            .write_image(f"{output_path}/{feat}_price_corr.png")


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    data, prices = load_data("../datasets/house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, prices, "./ex2Plots")
    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, prices)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    mean_loss = []
    std_loss = []
    for p in range(10, 101):
        current_loss = 0
        current_std = []
        for _ in range(10):
            Xtrain, ytrain, _1, _2 = split_train_test(train_X, train_y,
                                                      p / 100)
            loss = LinearRegression().fit(Xtrain.to_numpy(), ytrain.to_numpy()) \
                .loss(test_X.to_numpy(), test_y.to_numpy())
            current_loss += loss
            current_std.append(loss)

        mean_loss.append(current_loss / 10)
        std_loss.append(np.std(current_std))

    mean_loss, std_loss = np.array(mean_loss), np.array(std_loss)
    x_axis = np.arange(10, 101)
    confidence = lambda x: go.Scatter(x=x_axis,
                                      y=mean_loss + x * std_loss,
                                      name="confidence",
                                      fill="tonexty", mode="lines",
                                      line=dict(color="lightgrey"),
                                      showlegend=False)
    f = go.Figure(data=[go.Scatter(x=x_axis, y=mean_loss,
                                   name="avarage loss",
                                   mode="markers+lines",
                                   line=dict(dash="dash"),
                                   marker=dict(color="green", opacity=.7)
                                   ), confidence(2), confidence(-2)],
                  layout=go.Layout(
                      title="Average loss as function of training size with "
                            "error ribbon",
                      yaxis={"title": "Mean loss"},
                      xaxis={"title": "Training set size by percentage"})
                  )
    f.show()
