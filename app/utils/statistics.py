import pandas as pd
import numpy as np


def descriptive_stats(df, columns=None):
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not columns:
        return None
    stats = df[columns].describe().T
    stats['median'] = df[columns].median()
    stats['skewness'] = df[columns].skew()
    stats['kurtosis'] = df[columns].kurtosis()
    stats['missing'] = df[columns].isnull().sum()
    stats['missing_pct'] = round(df[columns].isnull().sum() / len(df) * 100, 2)
    stats = stats.rename(columns={
        'count': 'count', 'mean': 'mean', 'std': 'std',
        'min': 'min', '25%': 'q25', '50%': 'q50',
        '75%': 'q75', 'max': 'max'
    })
    stats = stats[['count', 'missing', 'missing_pct', 'mean', 'std', 'min', 'q25', 'median', 'q50', 'q75', 'max', 'skewness', 'kurtosis']]
    stats = stats.round(2)
    return stats.to_dict('index')


def categorical_stats(df, columns=None):
    if not columns:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    result = {}
    for col in columns:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        top = vc.index[0] if not vc.empty else None
        result[col] = {
            'unique_count': int(df[col].nunique()),
            'top_value': str(top),
            'top_freq': int(vc.iloc[0]) if not vc.empty else 0,
            'top_pct': round(vc.iloc[0] / len(df) * 100, 2) if not vc.empty else 0,
            'missing': int(df[col].isnull().sum()),
            'value_counts': {str(k): int(v) for k, v in vc.head(10).items()}
        }
    return result


def correlation_matrix(df, method='pearson', columns=None):
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(columns) < 2:
        return None
    corr = df[columns].corr(method=method)
    corr = corr.round(3)
    result = {
        'columns': list(corr.columns),
        'index': list(corr.index),
        'values': corr.values.tolist()
    }
    return result


def group_stats(df, group_col, value_cols=None, agg_funcs=None):
    if group_col not in df.columns:
        return None
    if not value_cols:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not agg_funcs:
        agg_funcs = ['count', 'mean', 'std', 'min', 'max']
    valid_funcs = [f for f in agg_funcs if f in ['count', 'mean', 'std', 'min', 'max', 'median', 'sum']]
    cols = [c for c in value_cols if c in df.columns and c != group_col]
    if not cols:
        return None
    grouped = df.groupby(group_col)[cols].agg(valid_funcs).round(2)
    grouped.columns = [f'{col}_{func}' for col, func in grouped.columns]
    data = {}
    for idx, row in grouped.iterrows():
        data[str(idx)] = row.to_dict()
    return {
        'group_column': group_col,
        'value_columns': cols,
        'agg_funcs': valid_funcs,
        'data': data
    }


def frequency_table(df, column):
    if column not in df.columns:
        return None
    vc = df[column].value_counts().reset_index()
    vc.columns = [column, '频数']
    vc['频率(%)'] = round(vc['频数'] / len(df) * 100, 2)
    vc['累计频率(%)'] = round(vc['频率(%)'].cumsum(), 2)
    return vc.to_dict('records')
