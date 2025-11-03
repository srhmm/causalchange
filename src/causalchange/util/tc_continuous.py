import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma, gamma
from math import log, pi


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CNST
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma, gamma
from math import log, pi


def conditional_mutual_information(X, Y, S1, S2, k=10):
    return conditional_mutual_information_resids(X, Y, S1, S2, k)

def entropy_knn(X, k=3):
    """Estimate the entropy of a random variable X using k-nearest neighbors."""
    n, d = X.shape
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nn.kneighbors(X)
    avg_log_dist = np.mean(np.log(distances[:, -1]))
    return digamma(n) - digamma(k) + d * avg_log_dist + log(pi ** (d / 2)) - log(gamma(d / 2 + 1))

def mutual_information_knn(X, Y, k=3):
    """Estimate the mutual information I(X; Y) using k-nearest neighbors."""
    XY = np.hstack([X, Y])
    H_X = entropy_knn(X, k=k)
    H_Y = entropy_knn(Y, k=k)
    H_XY = entropy_knn(XY, k=k)
    return max(0, H_X + H_Y - H_XY)

def compute_residuals(X, S):
    """Compute residuals by regressing S on X using Gaussian Process regression."""
    if S.shape[1] == 0:  # If S is empty, return X as is
        return X
    # TODO GPs:
    #kernel = CNST(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
    #gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    model = LinearRegression()
    model.fit(S, X)
    X_pred = model.predict(S)
    return X - X_pred

def conditional_mutual_information_resids(X, Y, S1, S2, k=3):
    """Estimate the conditional mutual information I(X; Y | S1, S2) using Gaussian Process residuals."""
    R_X = compute_residuals(X, S1)
    R_Y = compute_residuals(Y, S2)
    return mutual_information_knn(R_X, R_Y, k=k)
'''
# Generate synthetic data based on a simple causal graph
np.random.seed(42)
n_samples = 1000

# Variables in the causal graph
A = np.random.normal(0, 1, n_samples).reshape(-1, 1)
B = 0.5 * A + np.random.normal(0, 1, n_samples).reshape(-1, 1)
C = 0.5 * B + np.random.normal(0, 1, n_samples).reshape(-1, 1)
D = 0.5 * A + 0.5 * C + np.random.normal(0, 1, n_samples).reshape(-1, 1)

# Test cases using the new CMI implementation:
# 1. Compute CMI I(D; B | A, C) - correct parents
cmi_correct = conditional_mutual_information_resids(D, B, np.hstack([A, C]), np.hstack([A, C]))
print(f"CMI I(D; B | A, C): {cmi_correct:.4f}")

# 2. Compute CMI I(D; B) - no parents
cmi_unconditional = conditional_mutual_information_resids(D, B, np.array([]).reshape(n_samples, 0), np.array([]).reshape(n_samples, 0))
print(f"CMI I(D; B): {cmi_unconditional:.4f}")

# 3. Compute CMI I(D; B | A) - missing a correct parent (C)
cmi_missing_C = conditional_mutual_information_resids(D, B, A, A)
print(f"CMI I(D; B | A): {cmi_missing_C:.4f}")

# 4. Compute CMI I(D; B | C) - missing a correct parent (A)
cmi_missing_A = conditional_mutual_information_resids(D, B, C, C)
print(f"CMI I(D; B | C): {cmi_missing_A:.4f}")

# 5. Compute CMI I(D; B | wrong S) - using wrong conditioning set
S_wrong = np.hstack([A, np.random.normal(0, 1, (n_samples, 1))])
cmi_wrong = conditional_mutual_information_resids(D, B, S_wrong, S_wrong)
print(f"CMI I(D; B | wrong S): {cmi_wrong:.4f}")
 
def entropy_knn(X, k=3):
    """Estimate the entropy of a random variable X using k-nearest neighbors."""
    n, d = X.shape
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    avg_log_dist = np.mean(np.log(distances[:, -1]))
    return digamma(n) - digamma(k) + d * avg_log_dist + log(pi ** (d / 2)) - log(gamma(d / 2 + 1))


def mutual_information_knn(X, Y, k=3):
    """Estimate the mutual information I(X; Y) using k-nearest neighbors."""
    XY = np.hstack([X, Y])
    H_X = entropy_knn(X, k=k)
    H_Y = entropy_knn(Y, k=k)
    H_XY = entropy_knn(XY, k=k)
    return max(0, H_X + H_Y - H_XY)
'''

#todo S1, S2 seperately?
def conditional_mutual_information_knn(X, Y, S, k=10):
    """Estimate the conditional mutual information I(X; Y | S) using k-nearest neighbors."""

    XS = np.hstack([X, S])
    YS = np.hstack([Y, S])
    XYS = np.hstack([X, Y, S])

    H_XS = entropy_knn(XS, k=k)
    H_YS = entropy_knn(YS, k=k)
    H_S = entropy_knn(S, k=k) if S.shape[1] > 0 else 0 #todo does this make sense?
    H_XYS = entropy_knn(XYS, k=k)

    return max(0, H_XS + H_YS - H_S - H_XYS)

if False:
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # S1 and S2 as independent random variables
    S1 = np.random.normal(0, 1, (n_samples, 1))
    S2 = np.random.normal(0, 1, (n_samples, 1))

    # Introduce a dependency between S1 and S2 (optional)
    S2 = 0.5 * S1 + np.random.normal(0, 1, (n_samples, 1))

    # X depends on S1 and some noise
    X = S1 + np.random.normal(0, 0.5, (n_samples, 1))

    # Y depends on S2 and some noise
    Y = S2 + np.random.normal(0, 0.5, (n_samples, 1))

    # Estimate mutual information between P(X | S1) and P(Y | S2)
    mi_conditional = conditional_mutual_information_knn(X, Y, np.hstack([S1, S2]) )
    print(f"Mutual Information I(X; Y | S1, S2): {mi_conditional:.4f}")

    # Also, compare with unconditional mutual information
    mi_unconditional = mutual_information_knn(X, Y )
    print(f"Mutual Information I(X; Y): {mi_unconditional:.4f}")

