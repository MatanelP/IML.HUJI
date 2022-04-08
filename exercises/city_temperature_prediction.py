import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=["Date"])
    # remove duplicates and empty entries
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    # remove features that's being considered as errors
    data.drop(data.loc[data["Temp"] < -15].index, inplace=True)
    data.drop(data.loc[data["Day"] < 1].index, inplace=True)
    data.drop(data.loc[data["Day"] > 31].index, inplace=True)
    data.drop(data.loc[data["Month"] < 1].index, inplace=True)
    data.drop(data.loc[data["Month"] > 12].index, inplace=True)
    data.drop(data.loc[data["Year"] < 1950].index, inplace=True)
    data.drop(data.loc[data["Year"] > 2023].index, inplace=True)
    # adding DayOfYear
    data["DayOfYear"] = data["Date"].dt.dayofyear
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    data_israel = data.drop(data[data["Country"] != "Israel"].index)
    data_israel["Year"] = data_israel["Year"].astype(str)
    israel_temp_dayOfYear_plot = px.scatter(data_israel, "DayOfYear", "Temp",
                                            "Year",
                                            title="Tempeture in Israel as a function of day of year")
    israel_temp_dayOfYear_plot.show()

    months_std = data_israel.groupby(["Month"]).agg("std")
    israel_temp_dayOfYear_byMounth_plot = px.bar(months_std, months_std.index,
                                                 "Temp", labels={
            "Temp": "STD for Temp"},
                                                 title="Standard deviation of the daily temperature by month in Israel")
    israel_temp_dayOfYear_byMounth_plot.show()

    # Question 3 - Exploring differences between countries
    group_month_country_mean_std = data.groupby(["Country", "Month"])[
        "Temp"].agg(["mean", "std"]).reset_index()
    group_month_country_mean_std_plot = px.line(
        group_month_country_mean_std,
        "Month",
        "mean", error_y="std",
        title="Monthly temperature on average",
        labels={"mean": "average with deviation"},
        color="Country")
    group_month_country_mean_std_plot.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(
        data_israel["DayOfYear"], data_israel["Temp"])
    loss_for_k = []
    for k in range(1, 11):
        fitting = PolynomialFitting(k + 1)
        fitting.fit(train_X.to_numpy(), train_y.to_numpy())
        loss_for_k.append(
            round(fitting.loss(test_X.to_numpy(), test_y.to_numpy()), 2))
        print(loss_for_k[k - 1])
    loss_for_k_plot = px.bar(x=range(1, 11), y=loss_for_k,
                             title="Loss calculated by a degree of fitting",
                             labels={"x": "degree of fitting",
                                     "y": "loss calculate"})
    loss_for_k_plot.show()

    # Question 5 - Evaluating fitted model on different countries
    fitting_5 = PolynomialFitting(5)
    fitting_5.fit(data_israel["DayOfYear"], data_israel["Temp"])
    error = []
    all_countries = ["Jordan", "South Africa", "The Netherlands"]
    for country in all_countries:
        country_data = data[data["Country"] == country]
        error.append(
            fitting_5.loss(country_data["DayOfYear"], country_data["Temp"]))
    error_by_country_plot = px.bar(x=all_countries, y=error,
                                   labels={"x": "Country",
                                           "y": "Loss calculated"},
                                   title="Model’s (Poly, k = 5) error over"
                                         " each of the countries, fitted by"
                                         " Israel's Data")
    error_by_country_plot.show()
