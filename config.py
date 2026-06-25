import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'bigdata-secret-key-2024'
    UPLOAD_FOLDER = os.path.join(basedir, 'app', 'uploads')
    RESULT_FOLDER = os.path.join(basedir, 'app', 'results')
    DATASET_FOLDER = os.path.join(basedir, 'datasets')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
