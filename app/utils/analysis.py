import pandas as pd
import numpy as np


def analyze_file(file_path):

    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"File could not be read: {e}")

    if df.empty:
        raise ValueError("Uploaded file is empty")

    analysis = {}

    # --- Basic Info ---
    analysis["rows"]          = len(df)
    analysis["columns_count"] = len(df.columns)
    analysis["columns"]       = list(df.columns)
    analysis["duplicate_rows"]       = int(df.duplicated().sum())
    analysis["total_missing_cells"]  = int(df.isnull().sum().sum())
    analysis["memory_usage"]         = int(df.memory_usage(deep=True).sum())
    analysis["data_types"]           = df.dtypes.astype(str).to_dict()

    # --- Missing Values with percentage ---
    missing_count = df.isnull().sum().astype(int)
    missing_pct   = ((missing_count / len(df)) * 100).round(2)
    analysis["missing_values"] = {
        col: {"count": int(missing_count[col]), "percent": float(missing_pct[col])}
        for col in df.columns
    }

    # --- Numeric Analysis ---
    numeric_df = df.select_dtypes(include="number")
    analysis["numeric_columns"] = numeric_df.columns.tolist()

    if not numeric_df.empty:
        desc = numeric_df.describe()
        analysis["summary"] = desc.to_dict()

        # Outlier detection using IQR
        outliers = {}
        for col in numeric_df.columns:
            Q1  = numeric_df[col].quantile(0.25)
            Q3  = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = int(((numeric_df[col] < Q1 - 1.5 * IQR) |
                                  (numeric_df[col] > Q3 + 1.5 * IQR)).sum())
            outliers[col] = outlier_count
        analysis["outliers"] = outliers

        # Top correlated pairs
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr().abs()
            pairs = []
            cols  = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append({
                        "col1":        cols[i],
                        "col2":        cols[j],
                        "correlation": round(corr.iloc[i, j], 3)
                    })
            pairs = sorted(pairs, key=lambda x: x["correlation"], reverse=True)
            analysis["top_correlations"] = pairs[:5]

    # --- Categorical Analysis ---
    cat_df = df.select_dtypes(include="object")
    analysis["categorical_columns"] = cat_df.columns.tolist()
    cat_summary = {}
    for col in cat_df.columns:
        top_values = df[col].value_counts().head(5).to_dict()
        cat_summary[col] = {
            "unique_values": int(df[col].nunique()),
            "top_values":    {str(k): int(v) for k, v in top_values.items()}
        }
    analysis["categorical_summary"] = cat_summary

    # --- Smart Business Insights ---
    insights = []

    if analysis["duplicate_rows"] > 0:
        insights.append(f"⚠️ {analysis['duplicate_rows']} duplicate rows found — consider removing them.")

    for col, info in analysis["missing_values"].items():
        if info["percent"] > 30:
            insights.append(f"🔴 '{col}' has {info['percent']}% missing data — consider dropping this column.")
        elif info["percent"] > 10:
            insights.append(f"🟡 '{col}' has {info['percent']}% missing data — consider filling with mean/median.")

    if not numeric_df.empty:
        for col, count in analysis["outliers"].items():
            if count > 0:
                insights.append(f"📊 '{col}' has {count} outliers — may affect your analysis.")

        if "top_correlations" in analysis and analysis["top_correlations"]:
            top = analysis["top_correlations"][0]
            if top["correlation"] > 0.8:
                insights.append(f"🔗 Strong correlation ({top['correlation']}) between '{top['col1']}' and '{top['col2']}'.")

    analysis["insights"] = insights
    analysis["preview"]  = df.head(10).to_dict(orient="records")

    return analysis