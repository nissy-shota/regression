import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class Ridge_Regression:

    def __init__(self, alpha=1):
        self.w = None
        self.alpha = alpha

    def fit(self, X, y):
        
        X = np.insert(X, 0, 1, axis=1)
        I = np.eye(X.shape[1])
        self.w = np.linalg.inv(X.T @ X + self.alpha*I) @ X.T @ y
    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X @ self.w
        return y_pred

def main():

    # load data
    boston = load_boston()
    X, y = boston.data, boston.target
    features = boston.feature_names

    print(f'X shape {X.shape}')
    print(f'y shape {y.shape}')
    print(f'feature {features}')

    # StandardScaler
    X = StandardScaler().fit_transform(X)

    clf4 = Ridge_Regression(alpha=10)
    clf4.fit(X, y)

    coef_df = pd.DataFrame({'clf4': clf4.w[1:]},
                            index = boston.feature_names)

    
    plt.plot(clf4.w, label='alpha=10', color='b', linestyle='--')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Coefficient', fontsize=12)
    plt.legend();

    print(coef_df)



if __name__ == "__main__":
    main()