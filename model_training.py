# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation.evaluation_predictions import evaluate_predictions
from feature_engineering.transformer import LogTransformer

# %%

## ---------------------------------------------------------
## 3. Modelling
## ---------------------------------------------------------

## (a) Load and split cleaned data

df = pd.read_parquet("data/final_df.parquet")

df_train = df[df["year"] <= 2015].copy()
df_test = df[df["year"] > 2015].copy()

target = "Violent Rate"
feature_cols = [
    "Population",
    "Civilian_labor_force",
    "Unemployment_rate",
    "Traffic_Count",
    "Metro",
    "Urban_Influence_Code",
    "year",
]

X_train = df_train[feature_cols]
y_train = df_train[target]
X_test = df_test[feature_cols]
y_test = df_test[target]

# %%

## (b) Set Up Modelling Pipelines

numeric = [
    "Population",
    "Civilian_labor_force",
    "Unemployment_rate",
    "Traffic_Count",
    "Urban_Influence_Code",
    "year",
]

categorical = [
    "Metro",
]

log = ["Population", "Civilian_labor_force", "Traffic_Count"]

num_transform = Pipeline(
    steps=[("log", LogTransformer(columns=log)), ("scale", StandardScaler())]
)

cat_transform = OneHotEncoder(handle_unknown="ignore", drop="first")

preprocessor = ColumnTransformer(
    [
        ("num", num_transform, numeric),
        ("cat", cat_transform, categorical),
    ],
    remainder="drop",
)

# %%

# GLM Model Pipeline
tweedie = TweedieDistribution(1.5)

glm_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=tweedie, alpha=0.1, l1_ratio=0.0, fit_intercept=True
            ),
        ),
    ]
)

# %%

# LGBM Model Pipeline
lgbm_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5, random_state=42
            ),
        ),
    ]
)

# %%

# Fit the model
glm_pipeline.fit(X_train, y_train)
lgbm_pipeline.fit(X_train, y_train)
# %%

## ---------------------------------------------------------
## (c) Hyperparameter tuning

cv = KFold(n_splits=5, shuffle=True, random_state=42)

glm_param_grid = {
    "estimate__alpha": [0.001, 0.01, 0.1, 1.0],
    "estimate__l1_ratio": [0.0, 0.5, 1.0],
}

glm_grid = GridSearchCV(
    estimator=glm_pipeline,
    param_grid=glm_param_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2,
)

glm_grid.fit(X_train, y_train)

print("Best GLM params:", glm_grid.best_params_)

glm_best = glm_grid.best_estimator_
glm_test_pred = glm_best.predict(X_test)
glm_mse = mean_squared_error(y_test, glm_test_pred)
glm_rmse = np.sqrt(glm_mse)

print("GLM Test RMSE:", glm_rmse)

# %%

lgbm_param_grid = {
    "estimate__learning_rate": [0.01, 0.05, 0.1],
    "estimate__n_estimators": [300, 600, 1000],
    "estimate__num_leaves": [31, 50],
    "estimate__min_child_weight": [0.001, 0.01, 0.1],
}

lgbm_grid = GridSearchCV(
    estimator=lgbm_pipeline,
    param_grid=lgbm_param_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2,
)

lgbm_grid.fit(X_train, y_train)

print("Best LGBM params:", lgbm_grid.best_params_)

lgbm_best = lgbm_grid.best_estimator_
lgbm_test_pred = lgbm_best.predict(X_test)
lgbm_mse = mean_squared_error(y_test, lgbm_test_pred)
lgbm_rmse = np.sqrt(lgbm_mse)

print("LGBM Test RMSE:", lgbm_rmse)

# %%

## ---------------------------------------------------------
## 4. Evaluation and Interpretation
## ---------------------------------------------------------

# Evaluate predictions
df_test["GLM_pred"] = glm_best.predict(X_test)
df_test["LGBM_pred"] = lgbm_best.predict(X_test)

glm_eval = evaluate_predictions(
    df=df_test,
    outcome_column="Violent Rate",
    preds_column="GLM_pred",
    exposure_column=None,
)

lgbm_eval = evaluate_predictions(
    df=df_test,
    outcome_column="Violent Rate",
    preds_column="LGBM_pred",
    exposure_column=None,
)

print(glm_eval)
print(lgbm_eval)

# %%

## ---------------------------------------------------------
# Predicted vs. actual plot
def plot_pred_vs_actual(y_true, y_pred, title):
    """
    Plot a scatter plot (actual vs predicted value)
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("Actual Violent Rate")
    plt.ylabel("Predicted Violent Rate")
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_pred_vs_actual(y_test, df_test["GLM_pred"], "GLM: Predicted vs Actual")
plot_pred_vs_actual(y_test, df_test["LGBM_pred"], "LGBM: Predicted vs Actual")

# %%

## ---------------------------------------------------------
# Most relevant features
def feature_name(ct):
    """
    Get the name of feature variables after log transformation and one-hot encoding.
    Use for interpreting model coeffcient and feature importance.
    """
    output_features = []
    for name, transformer, columns in ct.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if isinstance(transformer, Pipeline):
            final_step = transformer.steps[-1][1]
            if hasattr(final_step, "get_feature_names_out"):
                names = final_step.get_feature_names_out(columns)
            else:
                names = columns
        elif hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(columns)
        else:
            names = columns
        output_features.extend(names)
    return output_features


feature_names = feature_name(glm_best.named_steps["preprocess"])
glm_coefs = glm_best.named_steps["estimate"].coef_

glm_importance = pd.DataFrame(
    {"feature": feature_names, "coefficient": glm_coefs}
).sort_values(by="coefficient", key=lambda s: s.abs(), ascending=False)

print(glm_importance.head(10))

lgbm_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": lgbm_best.named_steps["estimate"].feature_importances_,
    }
).sort_values(by="importance", ascending=False)

print(lgbm_importance.head(10))


# %%

## ---------------------------------------------------------
# Partial dependence plots
top5 = lgbm_importance["feature"].head(5).tolist()


for feat in top5:
    fig = plt.figure(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        lgbm_best, X_train, [feat], ax=plt.gca(), kind="average"
    )
    plt.title(f"PDP for {feat}")
    plt.tight_layout()
    plt.show()


# %%
