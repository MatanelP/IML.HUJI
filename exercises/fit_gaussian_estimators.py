from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    ug1 = UnivariateGaussian()
    samples = np.random.normal(10, 1, 1000)
    ug1.fit(samples)
    print((ug1.mu_, ug1.var_))

    # Question 2 - Empirically showing sample mean is consistent
    dists = []
    amount = []
    for m in range(10, samples.size + 1, 10):
        dists.append(m)
        ug1.fit(samples[:m])
        amount.append(abs(10 - ug1.mu_))

    go.Figure(go.Scatter(x=dists, y=amount, mode='markers+lines'),
              layout=go.Layout(
                  title=r"$\text{The absolute distance between the estimated"
                        r" and true value of the expectation, as a function of"
                        r" the sample size}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$\\text{distance - }|\hat\mu-\mu|$",
                  height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    ug1.fit(samples)
    go.Figure(go.Scatter(x=samples, y=ug1.pdf(samples), mode='markers'),
              layout=go.Layout(
                  title=r"$\text{The empirical sample PDF as a function of values}$",
                  xaxis_title="r$\\text{Values}$",
                  yaxis_title="r$\\text{PDF's}$",
                  height=500)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mg1 = MultivariateGaussian()
    mu = [0, 0, 4, 0]
    sigma = [[1, 0.2, 0, 0.5],
             [0.2, 2, 0, 0],
             [0, 0, 1, 0],
             [0.5, 0, 0, 1]]
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    mg1.fit(samples)
    print(mg1.mu_, mg1.cov_, sep='\n')

    # Question 5 - Likelihood evaluation
    f_1_3 = np.linspace(-10, 10, 200)
    size = 200
    map = [[mg1.log_likelihood(np.array((f_1_3[i], 0, f_1_3[j], 0)),
                               sigma, samples) for j in range(size)]
           for i in range(size)]
    go.Figure(go.Heatmap(x=f_1_3, y=f_1_3, z=map), layout=go.Layout(
        xaxis_title=r"$\text{Values for f3}$",
        yaxis_title=r"$\text{Values for f1}$",
        title=r"$\text{Log-Likelihood Expectation mu=[f1, 0, f3, 0]^T}$")).show()

    # Question 6 - Maximum likelihood
    f1 = f_1_3[np.argmax(map) // size]
    f3 = f_1_3[np.argmax(map) % size]
    print("f1: {:.3f}, f3: {:.3f}".format(f1, f3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
