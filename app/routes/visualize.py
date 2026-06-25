from flask import Blueprint, render_template, request, jsonify
from app.utils.data_loader import get_current_dataframe
from app.utils.statistics import correlation_matrix
from app.utils.visualization import (
    line_chart, bar_chart, scatter_plot, histogram, box_plot,
    heatmap, pie_chart, pair_plot, count_plot
)

viz_bp = Blueprint('visualize', __name__)


@viz_bp.route('/', methods=['GET', 'POST'])
def visualize():
    df, filename, error = get_current_dataframe()
    if error:
        return render_template('visualize.html', error=error)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    all_cols = df.columns.tolist()

    return render_template('visualize.html',
                           filename=filename,
                           numeric_cols=numeric_cols,
                           categorical_cols=categorical_cols,
                           all_cols=all_cols,
                           error=error)


@viz_bp.route('/chart', methods=['POST'])
def api_chart():
    df, _, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})

    chart_type = request.form.get('type', 'bar')
    x_col = request.form.get('x')
    y_col = request.form.get('y')
    y_cols = request.form.getlist('y_cols')
    color_col = request.form.get('color')
    title = request.form.get('title', '')

    fig = None
    if chart_type == 'line':
        if y_cols and x_col:
            fig = line_chart(df, x_col, y_cols, title)
    elif chart_type == 'bar':
        if x_col and y_col:
            fig = bar_chart(df, x_col, y_col, title)
    elif chart_type == 'scatter':
        if x_col and y_col:
            fig = scatter_plot(df, x_col, y_col, color_col, title=title)
    elif chart_type == 'histogram':
        if y_col:
            bins = int(request.form.get('bins', 20))
            fig = histogram(df, y_col, bins, title)
    elif chart_type == 'box':
        cols = y_cols if y_cols else ([y_col] if y_col else None)
        if cols:
            fig = box_plot(df, cols, x_col, title)
    elif chart_type == 'pie':
        if x_col and y_col:
            fig = pie_chart(df, x_col, y_col, title)
    elif chart_type == 'count':
        if x_col:
            fig = count_plot(df, x_col, title)
    elif chart_type == 'pair':
        cols = y_cols if y_cols else None
        if cols and len(cols) >= 2:
            fig = pair_plot(df, cols, color_col, title)
    elif chart_type == 'heatmap':
        method = request.form.get('corr_method', 'pearson')
        corr = correlation_matrix(df, method)
        if corr:
            fig = heatmap(corr, title)

    if not fig:
        return jsonify({'success': False, 'error': '图表生成失败，请检查参数'})
    return jsonify({'success': True, 'figure': fig})
