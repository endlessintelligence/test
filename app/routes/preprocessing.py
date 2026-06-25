import os
from flask import Blueprint, render_template, request, jsonify, session
from config import Config
from app.utils.data_loader import get_current_dataframe, get_upload_path, load_dataframe
from app.utils.preprocess import (
    handle_missing, detect_outliers_iqr, detect_outliers_zscore,
    remove_outliers, standardize, encode_categorical,
    get_missing_summary, get_basic_info
)

preproc_bp = Blueprint('preprocessing', __name__)

# 操作名映射（用于文件命名）
OP_NAMES = {
    'handle_missing': '缺失值处理',
    'remove_outliers': '移除异常值',
    'standardize': '标准化',
    'label_encode': '标签编码',
    'onehot_encode': '独热编码',
}


def _list_versions():
    original = session.get('original_file')
    current = session.get('current_file')
    if not original:
        return []
    root = os.path.splitext(original)[0]
    files = []
    for f in sorted(os.listdir(Config.UPLOAD_FOLDER)):
        if f.startswith(root) and f.endswith('.csv'):
            is_original = (f == original)
            is_current = (f == current)
            size = os.path.getsize(os.path.join(Config.UPLOAD_FOLDER, f))
            files.append({
                'name': f,
                'size': f'{size/1024:.1f} KB',
                'is_original': is_original,
                'is_current': is_current
            })
    return files


def _save_version(df, filename, op_name):
    name, ext = os.path.splitext(filename)
    new_name = f'{name}_{op_name}{ext}'
    path = get_upload_path(new_name)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    session['current_file'] = new_name
    return new_name


@preproc_bp.route('/', methods=['GET', 'POST'])
def preprocess():
    df, filename, error = get_current_dataframe()
    if error:
        return render_template('preprocessing.html', error=error, versions=[])

    basic_info = get_basic_info(df)
    missing_summary = get_missing_summary(df)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    all_cols = df.columns.tolist()
    preview = df.to_html(classes='table table-striped table-bordered table-sm', index=False)
    versions = _list_versions()
    original = session.get('original_file')

    return render_template('preprocessing.html',
                           filename=filename,
                           basic_info=basic_info,
                           missing_summary=missing_summary,
                           numeric_cols=numeric_cols,
                           categorical_cols=categorical_cols,
                           all_cols=all_cols,
                           preview=preview,
                           error=error,
                           versions=versions,
                           original_file=original)


@preproc_bp.route('/switch', methods=['POST'])
def switch_version():
    target = request.form.get('filename')
    path = get_upload_path(target)
    if not os.path.exists(path):
        return jsonify({'success': False, 'error': '文件不存在'})
    session['current_file'] = target
    return jsonify({'success': True})


@preproc_bp.route('/delete_version', methods=['POST'])
def delete_version():
    target = request.form.get('filename')
    original = session.get('original_file')
    if target == original:
        return jsonify({'success': False, 'error': '不能删除原始数据集'})
    path = get_upload_path(target)
    if os.path.exists(path):
        os.remove(path)
    if session.get('current_file') == target:
        session['current_file'] = original
    return jsonify({'success': True})


@preproc_bp.route('/clean', methods=['POST'])
def clean_versions():
    original = session.get('original_file')
    current = session.get('current_file')
    if not original:
        return jsonify({'success': False, 'error': '没有可清理的版本'})
    root = os.path.splitext(original)[0]
    deleted = 0
    for f in list(os.listdir(Config.UPLOAD_FOLDER)):
        if not (f.startswith(root) and f.endswith('.csv')):
            continue
        if f == original or f == current:
            continue
        os.remove(os.path.join(Config.UPLOAD_FOLDER, f))
        deleted += 1
    return jsonify({'success': True, 'deleted': deleted})


@preproc_bp.route('/handle_missing', methods=['POST'])
def api_handle_missing():
    df, filename, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    method = request.form.get('method', 'drop_row')
    columns = request.form.getlist('columns') or None
    fill_value = request.form.get('fill_value')
    df_result = handle_missing(df, method, columns, fill_value)
    _save_version(df_result, filename, OP_NAMES['handle_missing'])
    return jsonify({'success': True, 'rows': len(df_result), 'cols': len(df_result.columns)})


@preproc_bp.route('/detect_outliers', methods=['POST'])
def api_detect_outliers():
    df, filename, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    method = request.form.get('method', 'iqr')
    columns = request.form.getlist('columns') or None
    if method == 'iqr':
        factor = float(request.form.get('factor', 1.5))
        info = detect_outliers_iqr(df, columns, factor)
    else:
        threshold = float(request.form.get('threshold', 3))
        info = detect_outliers_zscore(df, columns, threshold)
    return jsonify({'success': True, 'outliers': info})


@preproc_bp.route('/remove_outliers', methods=['POST'])
def api_remove_outliers():
    df, filename, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    method = request.form.get('method', 'iqr')
    columns = request.form.getlist('columns') or None
    factor = float(request.form.get('factor', 1.5))
    threshold = float(request.form.get('threshold', 3))
    df_result = remove_outliers(df, method, columns, factor, threshold)
    _save_version(df_result, filename, OP_NAMES['remove_outliers'])
    return jsonify({'success': True, 'rows': len(df_result), 'cols': len(df_result.columns)})


@preproc_bp.route('/standardize', methods=['POST'])
def api_standardize():
    df, filename, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    method = request.form.get('method', 'standard')
    columns = request.form.getlist('columns') or None
    df_result = standardize(df, method, columns)
    _save_version(df_result, filename, OP_NAMES['standardize'])
    return jsonify({'success': True, 'rows': len(df_result), 'cols': len(df_result.columns)})


@preproc_bp.route('/encode', methods=['POST'])
def api_encode():
    df, filename, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})
    method = request.form.get('method', 'label')
    columns = request.form.getlist('columns') or None
    df_result, info = encode_categorical(df, method, columns)
    op = OP_NAMES['label_encode'] if method == 'label' else OP_NAMES['onehot_encode']
    _save_version(df_result, filename, op)
    return jsonify({'success': True, 'info': info, 'rows': len(df_result), 'cols': len(df_result.columns)})
