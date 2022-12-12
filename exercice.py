#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: Importez vos modules ici
import csv
from statistics import LinearRegression

import learn as learn
import numpy as np

from _exercice_version_prof import make_plot
import scikit-learn as sk


# TODO: Définissez vos fonctions ici
 def read_csv(path):
    df= pd.read_csv(path, sep=";", header=0)
    y = df["quality"]
    X = df.drop(columns=["quality"])
    return train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.1)
 def train_and_eval_model(model, X_train: list, X_test: list, y_train: list, y_test: list) -> list:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    make_plot(np.arange(len(y_test)), y_test, pred, model.__class__.__name__)
    return pred

 def make_plot(x: np.ndarray, target: list, pred: list, model_name: str) -> None:
    fig = plt.figure()
    plt.plot(x, target, label="Target values")
    from matplotlib import pyplot as plt
    plt.plot(x, pred, label="Predicted values")
    plt.legend()
    plt.title(f"{model_name} predictions analysis")
    plt.xlabel("Number of samples")
    plt.ylabel("Quality")
    fig.savefig(f"./{model_name}.png")




if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    X_train, X_test, y_train, y_test = read_csv()
    rf_pred = train_and_eval_model(RandomForestRegressor(), X_train, X_test, y_train, y_test)
    lr_pred = train_and_eval_model(LinearRegression(), X_train, X_test, y_train, y_test)
    print(f"Random Forest mse: {mean_squared_error(y_test, rf_pred)}")
    print(f"Linear regression mse: {mean_squared_error(y_test, lr_pred)}")

    pass