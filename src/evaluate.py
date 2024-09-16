import sys
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


def read_test_data(test_data_folder):
    X_test = pd.read_parquet(f'{test_data_folder}/X_test.parquet')
    y_test = pd.read_parquet(f'{test_data_folder}/y_test.parquet').iloc[:,0]

    print(X_test.shape, y_test.shape)

    return X_test, y_test


def evaluate_models(X_test, y_test, models, models_folder):
    for model_name, model in models.items():
        model = pickle.load(open(f'{models_folder}/{model_name}.pkl', "rb"))

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} accuracy: {accuracy}")


def main():

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py test-data-folder models-folder\n")
        sys.exit(1)

    X_test, y_test = read_test_data(sys.argv[1])

    models = {
        'LogisticRegression': LogisticRegression(),
        'DecisionTree': DecisionTreeClassifier(),
        'LGBM': LGBMClassifier()
    }

    evaluate_models(X_test, y_test, models, sys.argv[2])


if __name__ == "__main__":
    main()