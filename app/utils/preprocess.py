import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def handle_missing(df, method, columns=None, fill_value=None):
    df = df.copy()
    if columns:
        target_cols = [c for c in columns if c in df.columns]
    else:
        target_cols = list(df.columns)

    for col in target_cols:
        if method == 'drop_row':
            df = df.dropna(subset=[col])
        elif method == 'drop_col':
            df = df.drop(columns=[col])
        elif method == 'fill_mean':
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
        elif method == 'fill_median':
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
        elif method == 'fill_mode':
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else '')
        elif method == 'fill_value':
            df[col] = df[col].fillna(fill_value if fill_value else 0)
        elif method == 'fill_forward':
            df[col] = df[col].ffill()
        elif method == 'fill_backward':
            df[col] = df[col].bfill()
    return df


def detect_outliers_iqr(df, columns=None, factor=1.5):
    df = df.copy()
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_info = {}
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_info[col] = {
            'count': int(mask.sum()),
            'lower_bound': round(lower, 2),
            'upper_bound': round(upper, 2),
            'indices': df.index[mask].tolist()
        }
    return outlier_info


def detect_outliers_zscore(df, columns=None, threshold=3):
    df = df.copy()
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_info = {}
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        mask = z > threshold
        outlier_info[col] = {
            'count': int(mask.sum()),
            'threshold': threshold,
            'indices': df.index[mask].tolist()
        }
    return outlier_info


def remove_outliers(df, method='iqr', columns=None, factor=1.5, threshold=3):
    df = df.copy()
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            mask &= (df[col] >= lower) & (df[col] <= upper)
        elif method == 'zscore':
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask &= z <= threshold
    return df[mask]


def standardize(df, method='standard', columns=None):
    df = df.copy()
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        return df
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        return df
    df[valid_cols] = scaler.fit_transform(df[valid_cols])
    return df


def encode_categorical(df, method='label', columns=None):
    df = df.copy()
    if not columns:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    encoding_info = {}
    for col in columns:
        if col not in df.columns:
            continue
        if method == 'label':
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            encoding_info[col] = {'type': 'label', 'mapping': mapping}
        elif method == 'onehot':
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            encoding_info[col] = {'type': 'onehot', 'columns': list(dummies.columns)}
    return df, encoding_info


def get_missing_summary(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return []
    result = []
    for col, count in missing.items():
        result.append({
            'column': col,
            'missing_count': int(count),
            'missing_pct': round(count / len(df) * 100, 2)
        })
    return result


def get_basic_info(df):
    info = {
        'rows': len(df),
        'cols': len(df.columns),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'missing_cells': int(df.isnull().sum().sum()),
        'duplicated_rows': int(df.duplicated().sum()),
        'memory_usage': f'{df.memory_usage(deep=True).sum() / 1024:.1f} KB'
    }
    return info
