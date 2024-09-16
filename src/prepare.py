import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml


def read(path_file):
    df = pd.read_csv(path_file)
    return df


def hot_one_encode(df, cat_cols):
    # One Hot Encoding for categorical columns
    return pd.get_dummies(df, columns=cat_cols)


def scale(df, num_cols):
    # Standard Scaling for numerical columns
    std_scaler = StandardScaler()
    scaled_df = pd.DataFrame(std_scaler.fit_transform(df[num_cols].to_numpy()), columns=num_cols)
    df[num_cols] = scaled_df
    return df


def split(df, split_size, random_state):
    # Splitting the data into train and test sets
    X = df.drop('GradeClass', axis=1)
    y = df['GradeClass']
    return train_test_split(X, y, test_size=split_size, random_state=random_state)


def save(X_train, X_test, y_train, y_test, output_folder):
    # Saving the prepared data

    os.makedirs(output_folder, exist_ok=True)

    X_train.to_parquet(f'{output_folder}/X_train.parquet')
    X_test.to_parquet(f'{output_folder}/X_test.parquet')
    y_train.to_frame().to_parquet(f'{output_folder}/y_train.parquet')
    y_test.to_frame().to_parquet(f'{output_folder}/y_test.parquet')


def main():

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file output-folder\n")
        sys.exit(1)

    params = yaml.safe_load(open("params.yaml"))["prepare"]

    df = read(sys.argv[1])

    num_cols = ['StudyTimeWeekly', 'Absences', 'GPA']
    cat_cols = ['Sports', 'Volunteering', 'ParentalSupport', 'Music', 'Extracurricular', 'ParentalEducation', 'Age', 'Gender', 'Tutoring', 'Ethnicity']

    df = hot_one_encode(df, cat_cols)
    df = scale(df, num_cols)

    print(df.head())

    X_train, X_test, y_train, y_test = split(df, params["test_size"], params["random_state"])

    save(X_train, X_test, y_train, y_test, sys.argv[2])


if __name__ == "__main__":
    main()