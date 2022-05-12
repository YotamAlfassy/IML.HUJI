import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt


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
#     df = pd.read_csv('C:/Users/יותם אלפסי/Documents/GitHub/IML.HUJI/datasets/City_Temperature.csv',parse_dates=['Date'])
    df = pd.read_csv(filename,parse_dates=['Date'])
    df = df.fillna(0)

    df = df.loc[df['Year'] <2023] ## not future
    df = df.loc[df['Month'] <= 12] ## valid date
    df = df.loc[df['Month'] >= 1] ## valid date
    df = df.loc[df['Day'] <= 31] ## valid day
    df = df.loc[df['Day'] >= 1] ## valid day
    df = df.loc[df['Temp'] <= 55] ## valid Temp
    df = df.loc[df['Temp'] >= -40] ## valid Temp

    df['DayOfYear'] = df['Date'].dt.dayofyear

    ## irrelevant col: City
    df = df.drop(['City'], axis=1)

    return df
#     raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("C:/Users/יותם אלפסי/Documents/GitHub/IML.HUJI/datasets/City_Temperature.csv")
#     raise NotImplementedError()

    # Question 2 - Exploring data for specific country
    df = X
    df = df.loc[df['Country'] == "Israel"] ## israel
    fig, ax = plt.subplots()
    plt.scatter(df['DayOfYear'],df['Temp'],c=df['Year'])
    plt.title("Temp as a func of DayOfYear")
    plt.xlabel("Day of Year")
    plt.ylabel("Temp")
    plt.show()
    GroupByMonth = df.groupby(['Month'],as_index=False).agg({'Temp':'std'})
    ax = GroupByMonth.plot.bar(x='Month', y='Temp', rot=0,title="standard deviation by month",ylabel="Temp STD")
    plt.show()
#     raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    GroupByCountryAndMonth = X.groupby(['Country','Month']).agg({'Temp':['mean','std']})
    GroupByCountryAndMonth.columns = ["_".join(x) for x in GroupByCountryAndMonth.columns.ravel()]
    px.line(GroupByCountryAndMonth.reset_index(),
        x="Month",
        y="Temp_mean",
        color="Country",
        error_y = "Temp_std",
        title = "Average Monthly Temperature by Country, with std error"
    ).update_layout(xaxis={"type": "category"}).show()
#     raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    df = X.loc[X['Country'] == "Israel"] ## israel
    Y = df['Temp'].squeeze()
    train_x, train_y, test_x, test_y = split_train_test(df.drop(['Temp'], axis=1),Y)

    loss_array = []
    k_val = []
    for k in range(10):
        k_val.append(k+1)
        PF = PolynomialFitting(k+1)
        PF.fit(train_x['DayOfYear'].squeeze().to_numpy(),train_y.to_numpy())
        loss_array.append(round(PF.loss(test_x['DayOfYear'].squeeze().to_numpy(),test_y.to_numpy()),2))

    k_val = np.array(k_val)
    loss_array = np.array(loss_array)
    df_q4 = pd.DataFrame({'k_val':k_val,'loss':loss_array})
    print(df_q4)
    ax = df_q4.plot.bar(x='k_val', y='loss', rot=0,title="MSE by k degree",ylabel="MSE")
    plt.show()
#     raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    k = 3

    country_loss_array = []
    for country in X['Country'].unique():
        df_country = X.loc[X['Country'] == country]
        Y = df_country['Temp'].squeeze()
        train_x, train_y, test_x, test_y = split_train_test(df_country.drop(['Temp'], axis=1),Y)
        PF = PolynomialFitting(k)
        PF.fit(train_x['DayOfYear'].squeeze().to_numpy(),train_y.to_numpy())
        country_loss_array.append(round(PF.loss(test_x['DayOfYear'].squeeze().to_numpy(),test_y.to_numpy()),2))

    country_loss_array = np.array(country_loss_array)
    df_q5 = pd.DataFrame({'Country':X['Country'].unique(),f'loss_under_deg_{k}':country_loss_array})
    ax = df_q5.plot.bar(x='Country', y=f'loss_under_deg_{k}', rot=0,title=f"MSE by country on {k} degree poly fit",ylabel="MSE")
    plt.show()
#     raise NotImplementedError()