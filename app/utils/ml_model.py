import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import io


def prepare_data(df, feature_cols, target_col):
    """Clean and encode data automatically for any CSV"""
    df       = df.copy()
    encoders = {}

    df.dropna(subset=feature_cols + [target_col], inplace=True)

    for col in feature_cols + [target_col]:
        if df[col].dtype == object:
            # try cleaning and converting to number first
            cleaned = (df[col].astype(str)
                       .str.replace(",", "")
                       .str.replace(" ", "")
                       .str.replace("₹", "")
                       .str.replace("$", "")
                       .str.strip())
            try:
                df[col] = pd.to_numeric(cleaned)
            except (ValueError, TypeError):
                # truly categorical — label encode
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

    return df, encoders


def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img


def train_linear_model(file_path, feature_cols, target_col):
    """Train Linear Regression on any CSV — supports multiple features"""

    # load file
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    if len(df) < 10:
        raise ValueError("Dataset must have at least 10 rows.")

    # ensure feature_cols is a list
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    # prepare and clean data
    df, encoders = prepare_data(df, feature_cols, target_col)

    X = df[feature_cols].values
    y = df[target_col].values

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # metrics
    r2   = round(r2_score(y_test, y_pred), 4)
    mae  = round(mean_absolute_error(y_test, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

    # --- Chart 1: Actual vs Predicted ---
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.scatter(y_test, y_pred, alpha=0.6, color="#3b82f6", edgecolors="white", linewidth=0.5)
    ax1.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect fit")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted")
    ax1.legend()
    ax1.set_facecolor("#1e293b")
    fig1.patch.set_facecolor("#0f172a")
    ax1.tick_params(colors="#94a3b8")
    ax1.xaxis.label.set_color("#94a3b8")
    ax1.yaxis.label.set_color("#94a3b8")
    ax1.title.set_color("#f1f5f9")
    chart_actual_vs_pred = plot_to_base64(fig1)

    # --- Chart 2: Feature Importance ---
    coefficients = model.coef_
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = ["#22c55e" if c > 0 else "#ef4444" for c in coefficients]
    ax2.barh(feature_cols, coefficients, color=colors)
    ax2.set_xlabel("Coefficient Value")
    ax2.set_title("Feature Importance")
    ax2.axvline(0, color="white", linewidth=0.8)
    ax2.set_facecolor("#1e293b")
    fig2.patch.set_facecolor("#0f172a")
    ax2.tick_params(colors="#94a3b8")
    ax2.xaxis.label.set_color("#94a3b8")
    ax2.title.set_color("#f1f5f9")
    chart_feature_importance = plot_to_base64(fig2)

    # --- Chart 3: Residuals ---
    residuals = y_test - y_pred
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(residuals, bins=20, color="#f59e0b", edgecolor="#0f172a", alpha=0.9)
    ax3.set_xlabel("Residual (Actual - Predicted)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Residuals Distribution")
    ax3.set_facecolor("#1e293b")
    fig3.patch.set_facecolor("#0f172a")
    ax3.tick_params(colors="#94a3b8")
    ax3.xaxis.label.set_color("#94a3b8")
    ax3.yaxis.label.set_color("#94a3b8")
    ax3.title.set_color("#f1f5f9")
    chart_residuals = plot_to_base64(fig3)

    return {
        "metrics": {
            "r2_score":   r2,
            "mae":        mae,
            "rmse":       rmse,
            "train_size": len(X_train),
            "test_size":  len(X_test),
        },
        "feature_importance": dict(zip(feature_cols, [round(float(c), 4) for c in coefficients])),
        "charts": {
            "actual_vs_pred":      chart_actual_vs_pred,
            "feature_importance":  chart_feature_importance,
            "residuals":           chart_residuals,
        },
        "model":        model,
        "encoders":     encoders,
        "feature_cols": feature_cols,
        "target_col":   target_col,
        "accuracy":     r2,
    }


def predict_new(model_result, new_input_dict):
    """Predict on new user input"""
    model        = model_result["model"]
    encoders     = model_result["encoders"]
    feature_cols = model_result["feature_cols"]

    input_values = []
    for col in feature_cols:
        val = new_input_dict.get(col)
        if col in encoders:
            try:
                val = encoders[col].transform([str(val)])[0]
            except ValueError:
                val = 0
        try:
            input_values.append(float(str(val).replace(",", "").replace("₹", "").strip()))
        except (ValueError, TypeError):
            input_values.append(0.0)

    prediction = model.predict([input_values])[0]
    return round(float(prediction), 4)