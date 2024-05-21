import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    # Load the training and validation data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    X_train_list = X_train.tolist()
    y_train_list = y_train.tolist()
    X_val_list = X_val.tolist()
    y_val_list = y_val.tolist()

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        # Train the model
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train_list, y_train_list)
        y_pred = rf.predict(X_val_list)

        # Calculate RMSE
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE: {rmse}")

        # Log the RMSE metric to MLflow
        mlflow.log_metric("rmse", rmse)

        # Log the value of min_samples_split
        min_samples_split_value = rf.get_params()['min_samples_split']
        print(f"min samples split: ", min_samples_split_value)
        mlflow.log_param("min_samples_split", min_samples_split_value)


if __name__ == '__main__':
    run_train()