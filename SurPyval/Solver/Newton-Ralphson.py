import numpy as np
from numpy.linalg import solve, norm

def newton_ralphson(X, T, E, gradient_function, initial_beta=None, step_size=1.,
                     precision=10e-5, show_progress=True, include_likelihood=False):
    """
    Newton Rhaphson algorithm for fitting CPH model.
    Note that data is assumed to be sorted on T!
    Parameters:
        X: (n,d) Pandas DataFrame of observations.
        T: (n) Pandas Series representing observed durations.
        E: (n) Pandas Series representing death events.
        initial_beta: (1,d) numpy array of initial starting point for
                      NR algorithm. Default 0.
        step_size: float > 0.001 to determine a starting step size in NR algorithm.
        precision: the convergence halts if the norm of delta between
                 successive positions is less than epsilon.
        include_likelihood: saves the final log-likelihood to the CoxPHFitter under _log_likelihood.
    Returns:
        beta: (1,d) numpy array.
    """
    assert precision <= 1., "precision must be less than or equal to 1."
    n, d = X.shape

    # Want as bools
    E = E.astype(bool)

    # make sure betas are correct size.
    if initial_beta is not None:
        assert initial_beta.shape == (d, 1)
        beta = initial_beta
    else:
        beta = np.zeros((d, 1))

    get_gradients = gradient_function

    i = 1
    converging = True
    # 50 iterations steps with N-R is a lot.
    # Expected convergence is ~10 steps
    while i < 50:# and step_size > 0.001:

        output = get_gradients(X.values, beta, T.values, E.values, include_likelihood=include_likelihood)
        h, g = output[:2]

        delta = solve(-h, step_size * g.T)
        if np.any(np.isnan(delta)):
            raise ValueError("delta contains nan value(s). Convergence halted.")

        # Save these as pending result
        hessian, gradient = h, g

        if norm(delta) < precision:
            converging = False

        # Only allow small steps
        if norm(delta) > 10:
            step_size *= 0.5
            continue

        beta += delta

        if ((i % 10) == 0) and show_progress:
            print("Iteration %d: delta = %.5f" % (i, norm(delta)))
        i += 1

    return beta