"""
model.py
Trains and evaluates regression models to predict expected solar AC power output from weather metrics.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from src.features import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_DIR = Path(r"D:\solar panel underperformance analyser\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_expected_power_model(
    df: pd.DataFrame,
    plant_num: int = 1,
    save_model: bool = True
) -> tuple[lgb.LGBMRegressor, dict, pd.DataFrame]:
    """
    Trains a LightGBM regressor on operational daylight intervals to learn healthy nominal capacity.
    Returns:
        model: Trained LightGBM regressor
        metrics: Dictionary with R2, MAE, RMSE for train and test splits
        feature_importance_df: DataFrame of features and their importance values
    """
    # Filter for operational daylight hours so anomalies/failures don't pull the baseline down
    operational_mask = (df['IRRADIATION'] > 0.01) & (df['AC_POWER'] > 10.0)
    train_data = df[operational_mask].copy()

    X = train_data[FEATURE_COLUMNS]
    y = train_data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
    }

    feature_importance_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    if save_model:
        model_path = MODEL_DIR / f"expected_power_plant_{plant_num}.joblib"
        joblib.dump(model, model_path)

    return model, metrics, feature_importance_df


def load_or_train_model(df: pd.DataFrame, plant_num: int = 1) -> tuple[lgb.LGBMRegressor, dict, pd.DataFrame]:
    """
    Loads saved model if available, otherwise trains and saves a new one.
    """
    model_path = MODEL_DIR / f"expected_power_plant_{plant_num}.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        # Evaluate quick test metrics
        operational_mask = (df['IRRADIATION'] > 0.01) & (df['AC_POWER'] > 10.0)
        eval_data = df[operational_mask]
        X = eval_data[FEATURE_COLUMNS]
        y = eval_data[TARGET_COLUMN]
        y_pred = model.predict(X)
        metrics = {
            "test_r2": float(r2_score(y, y_pred)),
            "test_mae": float(mean_absolute_error(y, y_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        }
        feature_importance_df = pd.DataFrame({
            'Feature': FEATURE_COLUMNS,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        return model, metrics, feature_importance_df
    else:
        return train_expected_power_model(df, plant_num=plant_num, save_model=True)


def predict_expected_power(model: lgb.LGBMRegressor, df: pd.DataFrame) -> np.ndarray:
    """
    Predicts expected AC power. Enforces zero generation during night/zero-irradiation slots.
    """
    X = df[FEATURE_COLUMNS]
    predictions = model.predict(X)
    predictions = np.clip(predictions, a_min=0.0, a_max=None)

    # Physical boundary condition: zero power when irradiation is zero
    night_mask = (df['IRRADIATION'] <= 0.005)
    predictions[night_mask] = 0.0

    return predictions


if __name__ == "__main__":
    from src.data_loader import merge_and_clean_data
    from src.features import add_solar_features

    for p in [1, 2]:
        print(f"\nTraining model for Plant {p}...")
        raw_df = merge_and_clean_data(p)
        featured_df = add_solar_features(raw_df)
        mod, mets, fi = train_expected_power_model(featured_df, plant_num=p)
        print(f"Plant {p} Test R2: {mets['test_r2']:.4f}, MAE: {mets['test_mae']:.2f} kW, RMSE: {mets['test_rmse']:.2f} kW")
        print("Top 3 Features:\n", fi.head(3))
