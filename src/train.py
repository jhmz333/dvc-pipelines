import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import yaml


def read_train_data(input_folder):
    X_train = pd.read_parquet(f'{input_folder}/X_train.parquet')
    y_train = pd.read_parquet(f'{input_folder}/y_train.parquet').iloc[:,0]

    print(X_train.shape, y_train.shape)

    return X_train, y_train


def train_models(X_train, y_train, models):
    for model_name, model in models.items():
        model.fit(X_train, y_train)


def save_models(models, output_folder):
    for model_name, model in models.items():
        pickle.dump(model, open(f'{output_folder}/{model_name}.pkl', 'wb'))


def main():

    params = yaml.safe_load(open("config/params.yaml"))["train"]

    X_train, y_train = read_train_data(params["input_folder"])

    models = {
        'LogisticRegression': LogisticRegression(),
        'DecisionTree': DecisionTreeClassifier(),
        'LGBM': LGBMClassifier()
    }

    train_models(X_train, y_train, models)

    save_models(models, params["output_folder"])


if __name__ == "__main__":
    main()