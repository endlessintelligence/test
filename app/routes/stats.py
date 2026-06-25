import json
from flask import Blueprint, render_template, request, jsonify, session
from app.utils.data_loader import get_current_dataframe
from app.utils.statistics import (
    descriptive_stats, categorical_stats, correlation_matrix,
    group_stats, frequency_table
)

stats_bp = Blueprint('stats', __name__)


@stats_bp.route('/', methods=['GET', 'POST'])
def statistics():
    df, filename, error = get_current_dataframe()
    if error:
        return render_template('stats.html', error=error)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    all_cols = df.columns.tolist()

    return render_template('stats.html',
                           filename=filename,
                           numeric_cols=numeric_cols,
                           categorical_cols=categorical_cols,
                           all_cols=all_cols,
                           error=error)


@stats_bp.route('/descriptive', methods=['POST'])
def api_descriptive():
    df, _, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    columns = request.form.getlist('columns') or None
    result = descriptive_stats(df, columns)
    return jsonify({'success': True, 'data': result})


@stats_bp.route('/categorical', methods=['POST'])
def api_categorical():
    df, _, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    columns = request.form.getlist('columns') or None
    result = categorical_stats(df, columns)
    return jsonify({'success': True, 'data': result})


@stats_bp.route('/correlation', methods=['POST'])
def api_correlation():
    df, _, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    method = request.form.get('method', 'pearson')
    columns = request.form.getlist('columns') or None
    result = correlation_matrix(df, method, columns)
    return jsonify({'success': True, 'data': result})


@stats_bp.route('/group', methods=['POST'])
def api_group():
    df, _, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    group_col = request.form.get('group_col')
    value_cols = request.form.getlist('value_cols') or None
    agg_funcs = request.form.getlist('agg_funcs') or None
    result = group_stats(df, group_col, value_cols, agg_funcs)
    return jsonify({'success': True, 'data': result})


@stats_bp.route('/frequency', methods=['POST'])
def api_frequency():
    df, _, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    column = request.form.get('column')
    result = frequency_table(df, column)
    return jsonify({'success': True, 'data': result})
