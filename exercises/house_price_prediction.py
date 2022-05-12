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
#     df = pd.read_csv('C:/Users/יותם אלפסי/Documents/GitHub/IML.HUJI/datasets/house_prices.csv')
    df = pd.read_csv(filename)
    df = df.fillna(0)

    df = df.loc[df['price'] > 0] ## price is positive
    df = df.loc[df['sqft_lot15'] > 0] ## sqft_lot15 is positive
    df = df.loc[df['bedrooms'] > 0] ## bedrooms is positive
    df = df.loc[df['bedrooms'] != 33] ## remove 33 bedroom (probably typo)
    df = df.loc[df['bathrooms'] > 0] ## bathrooms is positive

    ## fix dates
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = pd.DatetimeIndex(df['date']).year
    df['date'] = df['date'].astype('int32')

    ## create var last year and was renovated
    df['last_year_update'] = df[['yr_built','yr_renovated']].max(axis=1).astype('int')
    df['was_renovated'] = np.where(df['yr_renovated'] > 0, 1, 0)
    
#     catagorical = ['view', 'condition', 'grade']
#     for i in catagorical:
#         d = pd.get_dummies(df[i])
#         for (columnName, columnData) in d.iteritems():
#             df[f'{i}_{columnName}']=columnData
    
    # catagorical view
    df['view_0'] = 0
    df['view_0'] = np.where(df['view'] == 0, 1, 0)
    df['view_1'] = 0
    df['view_1'] = np.where(df['view'] == 1, 1, 0)
    df['view_2'] = 0
    df['view_2'] = np.where(df['view'] == 2, 1, 0)
    df['view_3'] = 0
    df['view_3'] = np.where(df['view'] == 3, 1, 0)
    df['view_4'] = 0
    df['view_4'] = np.where(df['view'] == 4, 1, 0)

    ## catagorical condition (default is condition = 1)
    df['condition_1'] = 0
    df['condition_1'] = np.where(df['condition'] == 1, 1, 0)
    df['condition_2'] = 0
    df['condition_2'] = np.where(df['condition'] == 2, 1, 0)
    df['condition_3'] = 0
    df['condition_3'] = np.where(df['condition'] == 3, 1, 0)
    df['condition_4'] = 0
    df['condition_4'] = np.where(df['condition'] == 4, 1, 0)
    df['condition_5'] = 0
    df['condition_5'] = np.where(df['condition'] == 5, 1, 0)

    ## catagorical grade (default is grade = medium)
    df['grade_low'] = 0
    df['grade_low'] = np.where(df['grade'] < 4, 1, 0)
    df['grade_high'] = 0
    df['grade_high'] = np.where(df['grade'] > 10 , 1, 0)

    # vector response
#     Y = df['price'].to_numpy()
    Y = pd.DataFrame(df, columns = ['price'])
    Y = Y.squeeze()

    ## irrelevant col: ID, zipcode, yr_built, yr_renovated, view, condition, grade, price
    df = df.drop(['id','zipcode','yr_built','yr_renovated','view','condition','grade', 'price'], axis=1) 
    # X = df.to_numpy()

    return (df,Y)
#     raise NotImplementedError()


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
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
    
    for (columnName, columnData) in X.iteritems():
        ddd = pd.DataFrame({columnName:columnData, Y.name:Y})
        PC_val =  np.cov(columnData, Y, ddof=0)[1,0] / (np.std(columnData) * np.std(Y))
        ax = ddd.plot.scatter(x=columnName, y=Y.name,title= f"Pearson Correlation between {columnName} and response: {PC_val}\n")
        fig = ax.get_figure()
        fig.savefig(f"{output_path}{columnName}.pdf")
    
#     raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X,Y = load_data("C:/Users/יותם אלפסי/Documents/GitHub/IML.HUJI/datasets/house_prices.csv")
#     raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response
    X_Q2 = X[['sqft_living','view_1']]
    feature_evaluation(X_Q2,Y,"C:/Users/יותם אלפסי/Documents/")
#     raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X,Y)
#     raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mse_by_p_mean = []
    mse_by_p_var = []
    for i in range(10, 101):
        attepts_mse = []
        for attempt in range(10):
            x_sample = train_x.sample(frac=i/100)
            y_sample = train_y.iloc[train_y.index.isin(x_sample.index)]
            LR = LinearRegression()
            LR.fit(x_sample.to_numpy(),y_sample.to_numpy())
            attepts_mse.append(LR.loss(test_x.to_numpy(),test_y.to_numpy()))
        attepts_mse = np.array(attepts_mse)
        mse_by_p_mean.append(np.mean(attepts_mse))
        mse_by_p_var.append(attepts_mse.var(ddof=0))
    
    training_size = (np.arange(10,101,1)/100)*len(train_y)
    df = pd.DataFrame({'Avg MSE':mse_by_p_mean,'Training size':training_size})
    df['lower'] = np.array(mse_by_p_mean)-2*np.sqrt(np.array(mse_by_p_var))
    df['upper'] = np.array(mse_by_p_mean)+2*np.sqrt(np.array(mse_by_p_var))
    fig = go.Figure([go.Scatter(name='AVG MSE',x=df['Training size'],y=df['Avg MSE'],mode='lines',line=dict(color='rgb(31, 119, 180)'),),
                go.Scatter(name='Upper Bound', x=df['Training size'],y=df['upper'],mode='lines',marker=dict(color="#444"),line=dict(width=0),showlegend=False),
                go.Scatter(name='Lower Bound',x=df['Training size'],y=df['lower'],marker=dict(color="#444"),line=dict(width=0),mode='lines',fillcolor='rgba(68, 68, 68, 0.3)',fill='tonexty',showlegend=False)])
    fig.update_layout(yaxis_title='Avg MSE',xaxis_title='Training size',title='Plot average loss as function of training size',hovermode="x")
    fig.show()
#     raise NotImplementedError()
