from flask import Flask
from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    from app.routes.main import main_bp
    from app.routes.upload import upload_bp
    from app.routes.preprocessing import preproc_bp
    from app.routes.stats import stats_bp
    from app.routes.visualize import viz_bp
    from app.routes.ml import ml_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(preproc_bp, url_prefix='/preprocessing')
    app.register_blueprint(stats_bp, url_prefix='/stats')
    app.register_blueprint(viz_bp, url_prefix='/visualize')
    app.register_blueprint(ml_bp, url_prefix='/ml')

    return app
