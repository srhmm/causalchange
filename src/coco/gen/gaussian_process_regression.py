import numpy as np
from sklearn.linear_model import LinearRegression

from src.coco.gaussian_process_RFF import GaussianProcessRFF
from src.coco.gaussian_process_regularized import GaussianProcessRegularized


def lin_regression(X,y):
    return LinearRegression().fit(X, y)

#def gp_regression(X,y,rand_fourier_features):
#    gps = group_gaussianProcessRegression(np.array([X]),np.array([y]), rand_fourier_features=rand_fourier_features)
#    return gps[0]

def group_linearRegression(X, y, scale=False):
    """Given Data X,y in several contexts or groups, linear regression per group

    :param X: X, list of context/group data
    :param y: y
    :return: regression coefficient per context/group
    """
    coef_k_x = np.array([LinearRegression().fit(X[pi_k], y[pi_k]).coef_
                      for pi_k in range(len(X))])
    if scale:
        return coef_rescale(coef_k_x, len(X),  X_n=len(LinearRegression().fit(X[0], y[0]).coef_))

    return coef_k_x

"""
def group_gaussianProcessRegression(X, y,
                                    alpha=1.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2),
                                    scale=False, grid_search=False,
                                    rand_fourier_features=False,# True,
                                    pi_search=True,
                                    show_plt=False): 
    #alpha= (1e-1)
    kernel = 1 * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
    #kernel = RBFSampler(gamma=1, random_state=1)

    groups = range(len(X))
    GP_k = [None for _ in groups]
    grid_search = False
    for pi_k in groups:
        size_tr_local = len(X[pi_k])  # TODO use holdouts?
        tr_indices = np.sort(np.random.RandomState().choice(len(X[pi_k]), size=size_tr_local, replace=False))
        Xtr = X[pi_k][tr_indices]
        ytr = y[pi_k][tr_indices]

        if scale:
            Xtr = data_scale(Xtr)
            ytr = data_scale(ytr.reshape(-1,1))
        #TODO Optional: Grid search/CV for kernel parameter tuning

        param_grid = [{
            "alpha": [1e-2, 1e-3],
            "kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]
            }, {
            "alpha":  [1e-2, 1e-3],
            "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)] }]
        score = "r2" #or explained_variance

        if rand_fourier_features:
            regressor = GaussianProcessRFF() #
            regressor.fit(Xtr, ytr)
            GP_k[pi_k] = regressor
        elif grid_search:
            gaussianProcess = GaussianProcessRegularized()
            gaussianProcessGrid = GridSearchCV(estimator=gaussianProcess, param_grid=param_grid, cv=4,
                                           scoring='%s' % score)
            gaussianProcessGrid.fit(Xtr, ytr)
            GP_k[pi_k] = gaussianProcessGrid.best_estimator_
        else:
            gaussianProcess = GaussianProcessRegularized(kernel=kernel, alpha=alpha, n_restarts_optimizer=9)
            gaussianProcess.fit(Xtr, ytr)
            GP_k[pi_k] = gaussianProcess

        if show_plt:
            predictions = GP_k[pi_k].predict(X[pi_k])
            plt.scatter(X[pi_k], y[pi_k], label=str(pi_k) + " Values", linewidth=.2, marker=".")
            plt.scatter(X[pi_k],predictions,label= str(pi_k)+" Predict",linewidth=.4,marker = "+")
    if show_plt: plt.legend()
    return GP_k


def group_kernelRidgeRegression (X, y, show_plt=False):
    KR_k = [0 for _ in range(len(X))]

    for pi_k in range(len(X)): #range(len(Pi)):
        kernel_ridge = GridSearchCV(
            KernelRidge(kernel="rbf", gamma=0.1),
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
        )
        kernel_ridge.fit(X[pi_k], y[pi_k])
        KR_k[pi_k] = kernel_ridge
        if show_plt:
            predictions_kr = KR_k[pi_k].predict(X[pi_k])
            plt.scatter(X[pi_k], y[pi_k], label=str(pi_k)+ " Values", linewidth=.2, marker=".")
            plt.scatter(X[pi_k],predictions_kr,label=str(pi_k)+ " Predict (KR)",linewidth=.4,marker = "+")
    if show_plt: plt.legend()
    return(KR_k)

def group_SVRegression(X, y, show_plt):
    SVR_k = [0 for _ in range(len(X))]
    svr = GridSearchCV(
        SVR(kernel="rbf", gamma=0.1),
        param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
    )

    for pi_k in range(len(X)):
        svr.fit(X[pi_k], y[pi_k])
        SVR_k[pi_k] = svr
        if show_plt:
            predictions_kr = svr.predict(X[pi_k])
            plt.scatter(X[pi_k], y[pi_k], label=str(pi_k)+ " Values", linewidth=.2, marker=".")
            plt.scatter(X[pi_k],predictions_kr,label=str(pi_k) + " Predict (SVR)",linewidth=.4,marker = "+")
    if show_plt: plt.legend()
    return(SVR_k)
    """
def coef_rescale(coef_c_x, C_n, X_n):
    if X_n == 1:
        coef_c_x = coef_c_x.reshape(-1,1)
    #shape e.g. (5,2)
    else:
        c_n = coef_c_x.shape[0]
        x_n = coef_c_x.shape[1]
        assert x_n==X_n and  c_n == C_n
    return np.transpose([dim_rescale(coef_c_x[:, x_i]) for x_i in range(X_n)])

def dim_rescale(coef_c):
    offset = min(coef_c)
    coef_c = [coef_c[i] - offset for i in range(len(coef_c))]
    if (max(coef_c)==0): return([1 for i in range(len(coef_c))])

    scale = 100/max(coef_c)
    return [coef_c[i]* scale +1 for i in range(len(coef_c))]
