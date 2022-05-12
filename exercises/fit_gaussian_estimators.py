from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
# Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    size = 1000
    s = np.random.normal(mu,var,size)

    UG = UnivariateGaussian()
    UG.fit(s)
    print(f'Q3.1.1: (excpectation, variance): ({UG.mu_},{UG.var_})')
#     raise NotImplementedError()

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    samples_drawn = []
    ms = np.arange(10,1010,10)
    for m in ms:
        X = np.random.normal(mu, var, size=m)
        samples_drawn.append(X)
        UG.fit(X)
        estimated_mean.append(abs(UG.mu_-mu))

    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Q3.1.2: Abs(estimated_mu - real_mu) As Function Of Number Of Samples}$", 
                      xaxis_title="$m\\text{ - number of samples}$", 
                      yaxis_title="$\\text{abs(estimated_mu - real_mu)}$",
                      height=300)).show()
#     raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model
    i = 99
    m = 1000
    X = np.linspace(5, 15, m)
    Y = (1. / np.sqrt(2 * np.pi * var)) * np.exp(-(X - mu)**2 / (2 * var))
    pdf_instance = UG.pdf(s)

    go.Figure([go.Scatter(x=X, y=Y, mode='markers',line=dict(width=1, color="rgb(6,106,141)"), name=r'${True value}$'),
               go.Scatter(x=s, y=pdf_instance, mode='markers',line=dict(width=1, color="rgb(204,68,83)"), name=r'${sample}$'),],
              layout=go.Layout(title=r"$\text{Q3.1.3: PDF of sample, compared with real values}$", 
                      xaxis_title="$\\text{sample values}$", 
                      yaxis_title="$\\text{PDF}$",
                      height=500)).show()
#     raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    MG = MultivariateGaussian()
    size = 1000
    mean = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5],
           [0.2, 2, 0, 0],
           [0, 0, 1, 0],
           [0.5, 0, 0, 1]]
    X1 = np.random.multivariate_normal(mean, cov, size).T
    MG.fit(X1)
    print("Q3.2.4.1: Excpected Mean:\n",MG.mu_)
    print("Q3.2.4.2: Excpected Variance:\n",MG.cov_)
#     print(round(MG.cov_[0,0],3))
#     print(round(MG.cov_[0,3],3))
#     print(round(MG.mu_[1],3))
#     raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    mesh = np.array(np.meshgrid(f1, f3)).T.reshape(-1, 2)
    q5_Matrix = np.apply_along_axis(lambda i: MG.log_likelihood(np.array([i[0], 0, i[1], 0]), cov, X1), 1, mesh).reshape(200, 200)

    go.Figure(go.Heatmap(x=f1, y=f3, z=q5_Matrix), layout=go.Layout(title="Q3.2.5 - val of LL depending on f1&f3",
                xaxis_title="$\\text{f3 values}$", 
                yaxis_title="$\\text{f1 values}$", height=500, width=1000)).show()
#     raise NotImplementedError()

    # Question 6 - Maximum likelihood
    f1_max_val = round(f1[np.argmax(q5_Matrix,axis=0)[0]],3)
    f3_max_val = round(f3[np.argmax(q5_Matrix,axis=1)[0]],3)
    print(f'Q3.2.6: "f1: {f1_max_val}, f3: {f3_max_val} = {np.max(q5_Matrix)}"')
#     raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
