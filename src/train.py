"""Import modules"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error,
                             explained_variance_score,
                             mean_squared_log_error,
                             median_absolute_error,
                             max_error)
from xgboost import XGBRegressor
import mlflow
import mlflow.xgboost as xg


def main(): 
    
    """Main function"""
    # input and output argument
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, required=False, default=0.1)
    parser.add_argument("--n_estimator", type=int, required=False, default=100)
    parser.add_argument("--registered_model_name", type=str, help="model name")

    args = parser.parse_args()

    # start logging
    mlflow.start_run()

    # enable autologging
    xg.autolog()

    # prepare the data
    # print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    # print("input data: ", args.data)

    # Load the Diabetes dataset
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_train_ratio, random_state=42
    )

    # train the model

    # Initialize and train the XGBoost regressor
    model = XGBRegressor(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        objective='reg:squarederror',  # Use squared error for regression
    )

    # logs params
    mlflow.log_params(dict(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        objective='reg:squarederror'
    ))

    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance using multiple metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    explained_var_score = explained_variance_score(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred)
    med_abs_error = median_absolute_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    # log metrics
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('explained_var_score', explained_var_score)
    mlflow.log_metric('msle', msle)
    mlflow.log_metric('med_abs_error', med_abs_error)
    mlflow.log_metric('max_err', max_err)

    # Print the evaluation metrics
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae)
    print("Explained Variance Score:", explained_var_score)
    print("Mean Squared Log Error:", msle)
    print("Median Absolute Error:", med_abs_error)
    print("Max Error:", max_err)

    # save and register model
    # register
    print("Registering model via MLFlow...")
    xg.log_model(
        xgb_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name
    )

    # save the model
    xg.save_model(
        xgb_model=model,
        path=os.path.join(args.registered_model_name, "trained_model")
    )

    # Stop loggin
    mlflow.end_run()


if __name__ == "__main__":
    main()
