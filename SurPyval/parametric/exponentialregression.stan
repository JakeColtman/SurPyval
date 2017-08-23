data{
    int<lower=0> n_observations;
    int<lower=0> n_covariates
    real<lower=0> y_s[n_observations];
    matrix x_s[n_observations, n_covariates];
    real m_0[n_covariates];
    real variance_0[n_covariates, n_covariates];
}
parameters{
    real beta[n_covariates];
    real lambda[n_observations];
}
model{
    beta ~ normal(m_0, variance_0);
    lambda = exp(x_s * beta);
    y_s ~ exponential(lambda);
}
