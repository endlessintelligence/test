import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import json


def _fig_to_json(fig):
    return json.loads(fig.to_json())


def line_chart(df, x_col, y_cols, title='折线图'):
    fig = px.line(df, x=x_col, y=y_cols, title=title, markers=True)
    return _fig_to_json(fig)


def bar_chart(df, x_col, y_col, title='柱状图', orientation='v'):
    if orientation == 'h':
        fig = px.bar(df, y=x_col, x=y_col, title=title, orientation='h')
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title)
    return _fig_to_json(fig)


def scatter_plot(df, x_col, y_col, color_col=None, size_col=None, title='散点图'):
    df = df.dropna(subset=[x_col, y_col])
    kwargs = dict(x=x_col, y=y_col, title=title, opacity=0.7)
    if color_col:
        kwargs['color'] = color_col
    if size_col:
        kwargs['size'] = size_col
    fig = px.scatter(df, **kwargs)
    return _fig_to_json(fig)


def histogram(df, col, bins=20, title='直方图'):
    df = df.dropna(subset=[col])
    fig = px.histogram(df, x=col, nbins=bins, title=title, marginal='box')
    return _fig_to_json(fig)


def box_plot(df, y_cols, x_col=None, title='箱线图'):
    cols = y_cols + ([x_col] if x_col else [])
    df = df.dropna(subset=cols)
    if x_col:
        fig = px.box(df, x=x_col, y=y_cols, title=title)
    else:
        fig = px.box(df, y=y_cols, title=title)
    return _fig_to_json(fig)


def heatmap(corr_data, title='热力图'):
    if not corr_data or 'values' not in corr_data:
        return None
    z = corr_data['values']
    x = corr_data['columns']
    y = corr_data['index']
    fig = ff.create_annotated_heatmap(
        z, x=x, y=y,
        colorscale='RdBu_r',
        showscale=True,
        annotation_text=[[f'{v:.2f}' for v in row] for row in z]
    )
    fig.update_layout(title=title, width=600, height=500)
    return _fig_to_json(fig)


def pie_chart(df, names_col, values_col, title='饼图'):
    fig = px.pie(df, names=names_col, values=values_col, title=title)
    return _fig_to_json(fig)


def pair_plot(df, cols, color_col=None, title='散点矩阵图'):
    if len(cols) < 2 or len(cols) > 6:
        return None
    kwargs = dict(dimensions=cols, title=title)
    if color_col:
        kwargs['color'] = color_col
    fig = px.scatter_matrix(df, **kwargs)
    fig.update_layout(width=800, height=800)
    return _fig_to_json(fig)


def count_plot(df, col, title='计数图'):
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, 'count']
    fig = px.bar(vc, x=col, y='count', title=title)
    return _fig_to_json(fig)
