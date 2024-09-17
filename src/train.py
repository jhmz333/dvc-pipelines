import os
import sys
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


def read_train_data(input_folder):
    X_train = pd.read_parquet(f'{input_folder}/X_train.parquet')
    y_train = pd.read_parquet(f'{input_folder}/y_train.parquet').iloc[:,0]

    return X_train, y_train


def train_models(X_train, y_train, models):
    for model_name, model in models.items():
        model.fit(X_train, y_train)


def save_models(models, models_folder):

    os.makedirs(models_folder, exist_ok=True)

    for model_name, model in models.items():
        pickle.dump(model, open(f'{models_folder}/{model_name}.pkl', 'wb'))


def main():

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py input-folder models-folder\n")
        sys.exit(1)

    X_train, y_train = read_train_data(sys.argv[1])

    models = {
        'LogisticRegression': LogisticRegression(),
        'DecisionTree': DecisionTreeClassifier(),
        'LGBM': LGBMClassifier()
    }

    train_models(X_train, y_train, models)

    save_models(models, sys.argv[2])


if __name__ == "__main__":
    main()