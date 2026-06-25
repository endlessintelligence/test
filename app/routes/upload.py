import os
from flask import Blueprint, render_template, request, jsonify, session, send_from_directory
from config import Config
from app.utils.data_loader import (
    allowed_file, save_upload, load_builtin_dataset,
    get_builtin_datasets, get_current_dataframe, get_upload_path
)

upload_bp = Blueprint('upload', __name__)


@upload_bp.route('/', methods=['GET', 'POST'])
def upload_file():
    builtin_datasets = get_builtin_datasets()
    current_file = session.get('current_file')
    original_file = session.get('original_file')
    preview_data = None
    columns = []
    records_count = 0

    if current_file:
        df, filename, error = get_current_dataframe()
        if df is not None:
            columns = list(df.columns)
            records_count = len(df)
            preview_data = df.head(10).to_html(classes='table table-striped table-bordered table-hover', index=False)

    file_list = get_uploaded_files()
    rel_path = os.path.relpath(Config.UPLOAD_FOLDER, start=os.path.dirname(Config.RESULT_FOLDER))

    return render_template('upload.html',
                           builtin_datasets=builtin_datasets,
                           current_file=current_file,
                           original_file=original_file,
                           columns=columns,
                           records_count=records_count,
                           preview_data=preview_data,
                           file_list=file_list,
                           upload_folder=rel_path)


@upload_bp.route('/upload_file', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有选择文件'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'})
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '仅支持 CSV、Excel 文件'})
    try:
        filepath = save_upload(file)
        session['current_file'] = file.filename
        session['original_file'] = file.filename
        return jsonify({'success': True, 'filename': file.filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@upload_bp.route('/load_dataset', methods=['POST'])
def load_dataset():
    dataset_id = request.form.get('dataset')
    if not dataset_id:
        return jsonify({'success': False, 'error': '未指定数据集'})
    dst_name, error = load_builtin_dataset(dataset_id)
    if error:
        return jsonify({'success': False, 'error': error})
    session['current_file'] = dst_name
    session['original_file'] = dst_name
    return jsonify({'success': True, 'filename': dst_name})


@upload_bp.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename, as_attachment=True)


@upload_bp.route('/files')
def list_files():
    return jsonify({'files': get_uploaded_files()})


@upload_bp.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    path = get_upload_path(filename)
    if not os.path.exists(path):
        return jsonify({'success': False, 'error': '文件不存在'})
    os.remove(path)
    if session.get('current_file') == filename:
        session.pop('current_file', None)
    return jsonify({'success': True})


def get_uploaded_files():
    files = []
    if not os.path.exists(Config.UPLOAD_FOLDER):
        return files
    for f in sorted(os.listdir(Config.UPLOAD_FOLDER)):
        fpath = os.path.join(Config.UPLOAD_FOLDER, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            files.append({
                'name': f,
                'size': f'{size/1024:.1f} KB',
                'size_bytes': size,
                'mtime': os.path.getmtime(fpath)
            })
    return files
