import os
import pandas as pd
import shutil
from flask import session
from config import Config


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def load_dataframe(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        return pd.read_csv(filepath, encoding='utf-8-sig')
    elif ext in ('xlsx', 'xls'):
        return pd.read_excel(filepath)
    return None


def get_upload_path(filename):
    return os.path.join(Config.UPLOAD_FOLDER, filename)


def get_result_path(filename):
    return os.path.join(Config.RESULT_FOLDER, filename)


def save_upload(file):
    filepath = get_upload_path(file.filename)
    file.save(filepath)
    return filepath


def load_builtin_dataset(name):
    mapping = {
        'waimai': '外卖数据分析.csv',
        'grades': '学生成绩分析.csv',
        'consumption': '学生消费行为.csv',
        'course_eval': '课程评价数据.csv',
        'campus_card': '校园卡消费.csv',
    }
    if name not in mapping:
        return None, '数据集不存在'
    src = os.path.join(Config.DATASET_FOLDER, mapping[name])
    if not os.path.exists(src):
        return None, '数据集文件不存在'
    dst_name = mapping[name]
    dst = get_upload_path(dst_name)
    shutil.copy2(src, dst)
    return dst_name, None


def get_builtin_datasets():
    return [
        {'id': 'waimai', 'name': '外卖数据分析', 'desc': '大学生外卖订单数据，含价格、评分、配送时间等', 'records': '~200条', 'file': '外卖数据分析.csv'},
        {'id': 'grades', 'name': '学生成绩分析', 'desc': '多科目成绩数据，含平时分、期末分、总评等', 'records': '~150条', 'file': '学生成绩分析.csv'},
        {'id': 'consumption', 'name': '学生消费行为', 'desc': '校园一卡通消费记录，含消费金额、时间、类别', 'records': '~300条', 'file': '学生消费行为.csv'},
        {'id': 'course_eval', 'name': '课程评价数据', 'desc': '课程评分与评价数据，含教学质量多维评分', 'records': '~120条', 'file': '课程评价数据.csv'},
        {'id': 'campus_card', 'name': '校园卡消费', 'desc': '校园卡详细消费流水，商户类别、时段分析', 'records': '~500条', 'file': '校园卡消费.csv'},
    ]


def get_current_dataframe():
    filename = session.get('current_file')
    if not filename:
        return None, None, '未选择数据'
    filepath = get_upload_path(filename)
    if not os.path.exists(filepath):
        return None, None, '数据文件不存在'
    df = load_dataframe(filepath)
    return df, filename, None
