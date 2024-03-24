
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression


def debias_features(Xs_np, Xs_p, lambda_ = 1):
    """
    Function that applies the proposed geometric method to removes correlations between data and any number of protected variables.
    by He et al. (https://doi.org/10.1145/3375627.3375864)

    """
    assert Xs_np.shape[0]==Xs_p.shape[0]
    
    # Find orthonormal basis of protected features
    orthbasis = scipy.linalg.orth(Xs_p) #px

    # Debias nonprotected features
    Xs_np_debiased = Xs_np - orthbasis @ orthbasis.T @ Xs_np # rj

    # Return debiased nonprotected features
    return Xs_np_debiased + lambda_ * (Xs_np - Xs_np_debiased)


def evaluate_lambda(model, X_train_np, X_train_p, y_train, protected_feat_dict, lambda_range = [0, 1], load_computations = True, seed = 11012008):
    """
    Function to evaluate the effect of different lambdas in debiasiang of a dataset using the geometric method proposed by He et al. (https://doi.org/10.1145/3375627.3375864)
    Arguments:
     - model: The sklearn estimator to evaluate
     - X_train_np: The non-protected features
     - X_train_p: The protected features
     - y_train: The true labels for training
     - protected_feat_dict: Dictionary of protected features in the form <i: feat>, where i is in the index of the feature in X_p and feat is the feature string denomation.
    """
    # Collect the metrics of interest
    results = {"lambda": [], "accuracy": [], "f1": []}
    results.update({f"corr_{f}": [] for f in protected_feat_dict.values()})

    # Apply the different lambdas to the geometric method
    for l in np.linspace(lambda_range[0], lambda_range[1], 11):
        print(f"Trying lambda {l}...")
        # If we have saved the computed debiased features, load them

        # Acumulators for each metric
        acum_acc = []
        acum_f1 = []
        acum_corr = {f"corr_{f}": [] for f in protected_feat_dict.values()}

        # Partition the training set for cross-validation
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)

        # For each partition, we fit the model and calculate metrics
        for i, (train_index, test_index) in enumerate(kf.split(X_train_np)):
            print(f"Fold: {i+1}")
            # Calculate the debiased features for the partition
            X_np_deb = debias_features(X_train_np[train_index], X_train_p[train_index], lambda_ = l)
            # Fit the model with the debiased features
            model.fit(X_np_deb, y_train[train_index])
            # Predict on the testing part of the fold
            y_hat = model.predict(X_train_np[test_index])

            acc = accuracy_score(y_train[test_index], y_hat)
            f1 = f1_score(y_train[test_index], y_hat)

            print(f"Accuracy: {acc}")
            print(f"F1: {f1}")

            # Add the metrics for the fold
            acum_acc.append(acc)
            acum_f1.append(f1)
            
            # Include the correlation metrics between features
            for i, feat in protected_feat_dict.items():
                acum_corr[f"corr_{feat}"].append(compute_correlation_weight(X_np_deb, X_train_p[train_index, i].reshape(-1,1)))

        # Register the metrics for the given lambda (mean of all k-folds)
        results["lambda"].append(l)
        results["f1"].append(np.mean(acum_f1))
        results["accuracy"].append(np.mean(acum_acc))

        for i, feat in protected_feat_dict.items():
            results[f"corr_{feat}"].append(np.mean(acum_corr[f"corr_{feat}"]))
        
    # Returns the results as a DataFrame
    return pd.DataFrame(results).fillna(0)


def compute_correlation_weight(X_np, X_p, alpha = 0.05):
    """
    Function to compute the correlation between a set of not protected and protected features.
    It uses a custom metric:
        The average of the absolute value of the Pearson's correlation between all pairs of Non-Protected vs Protected features.
        Only significant values are considered.
    """

    # Array of correlations    
    corrs = []

    # Compute Pearson correlation between each pair of Non-Protected - Protected features
    for i in range(X_np.shape[1]):
        for j in range(X_p.shape[1]):
            # Correlation of the non-protected feature i with protected feature j
            c, p = pearsonr(X_np[:,i], X_p[:,j])
            # If significant, append the absolute correlation value (to avoid cancelling out)
            if p < alpha:
                corrs.append(abs(c))

    # Return the average of the correlations
    return np.mean(corrs)


model = LogisticRegression(penalty='l2', max_iter=1000)

protect_dict = {0: 'SEX', 1: 'RACE'}

Xs_train_l_np = np.load("./Xs_train_np.npy")
Xs_train_l_p = np.load("./Xs_train_p.npy")
y_train_l = np.load("./y_train.npy")

lambda_results = evaluate_lambda(model = model, X_train_np = Xs_train_l_np, X_train_p = Xs_train_l_p, y_train = y_train_l, 
                protected_feat_dict = protect_dict, lambda_range = [0, 1])
