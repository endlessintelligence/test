from flask import Blueprint, render_template, request, jsonify
from app.utils.data_loader import get_current_dataframe
from app.utils.preprocess import get_basic_info
from app.utils.ml_models import (
    kmeans_clustering, naive_bayes, neural_network, svm_classify,
    decision_tree, feature_selection, knn_classify, ridge_regression,
    predict_with_model, MLInputError
)

ml_bp = Blueprint('ml', __name__)

# 内存中保存当前模型（用于预测）
_current_model = None


@ml_bp.route('/', methods=['GET', 'POST'])
def ml():
    df, filename, error = get_current_dataframe()
    if error:
        return render_template('ml.html', error=error)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    all_cols = df.columns.tolist()
    basic_info = get_basic_info(df)
    preview = df.head(8).to_html(classes='table table-striped table-bordered table-sm', index=False)

    return render_template('ml.html',
                           filename=filename,
                           numeric_cols=numeric_cols,
                           categorical_cols=categorical_cols,
                           all_cols=all_cols,
                           error=error,
                           basic_info=basic_info,
                           preview=preview)


@ml_bp.route('/run', methods=['POST'])
def api_run():
    global _current_model
    df, _, error = get_current_dataframe()
    if error:
        return jsonify({'success': False, 'error': error})

    algorithm = request.form.get('algorithm')
    feature_cols = request.form.getlist('features')
    target_col = request.form.get('target')

    if not feature_cols:
        return jsonify({'success': False, 'error': '请选择特征列'})

    kwargs = {'df': df, 'feature_cols': feature_cols}

    try:
        result = None
        model_data = None

        if algorithm == 'kmeans':
            kwargs['n_clusters'] = int(request.form.get('n_clusters', 3))
            result = kmeans_clustering(**kwargs)
        elif algorithm == 'naive_bayes':
            if not target_col: return jsonify({'success': False, 'error': '请选择目标列'})
            kwargs.update(target_col=target_col, test_size=float(request.form.get('test_size', 0.3)))
            result, model_data = naive_bayes(**kwargs)
        elif algorithm == 'neural_network':
            if not target_col: return jsonify({'success': False, 'error': '请选择目标列'})
            kwargs.update(target_col=target_col, test_size=float(request.form.get('test_size', 0.3)),
                          hidden_layers=request.form.get('hidden_layers', '100,50'),
                          max_iter=int(request.form.get('max_iter', 500)),
                          activation=request.form.get('activation', 'relu'))
            result, model_data = neural_network(**kwargs)
        elif algorithm == 'svm':
            if not target_col: return jsonify({'success': False, 'error': '请选择目标列'})
            kwargs.update(target_col=target_col, test_size=float(request.form.get('test_size', 0.3)),
                          kernel=request.form.get('kernel', 'rbf'), C=float(request.form.get('C', 1.0)),
                          gamma=request.form.get('gamma', 'scale'))
            result, model_data = svm_classify(**kwargs)
        elif algorithm == 'decision_tree':
            if not target_col: return jsonify({'success': False, 'error': '请选择目标列'})
            kwargs.update(target_col=target_col, test_size=float(request.form.get('test_size', 0.3)),
                          min_samples_split=int(request.form.get('min_samples_split', 2)),
                          criterion=request.form.get('criterion', 'gini'))
            if request.form.get('max_depth'):
                kwargs['max_depth'] = int(request.form.get('max_depth'))
            result, model_data = decision_tree(**kwargs)
        elif algorithm == 'feature_selection':
            if not target_col: return jsonify({'success': False, 'error': '请选择目标列'})
            kwargs.update(target_col=target_col, k=int(request.form.get('k', 5)),
                          method=request.form.get('fs_method', 'f_classif'))
            result, model_data = feature_selection(**kwargs)
        elif algorithm == 'knn':
            if not target_col: return jsonify({'success': False, 'error': '请选择目标列'})
            kwargs.update(target_col=target_col, test_size=float(request.form.get('test_size', 0.3)),
                          n_neighbors=int(request.form.get('n_neighbors', 5)),
                          weights=request.form.get('weights', 'distance'),
                          metric=request.form.get('metric', 'euclidean'))
            result, model_data = knn_classify(**kwargs)
        elif algorithm == 'ridge':
            if not target_col: return jsonify({'success': False, 'error': '请选择目标列'})
            kwargs.update(target_col=target_col, test_size=float(request.form.get('test_size', 0.3)),
                          alpha=float(request.form.get('alpha', 1.0)))
            result, model_data = ridge_regression(**kwargs)
        else:
            return jsonify({'success': False, 'error': f'未知算法: {algorithm}'})

        if result is None:
            return jsonify({'success': False, 'error': '运行失败'})

        if model_data:
            _current_model = model_data

        return jsonify({'success': True, 'result': result})

    except MLInputError as e:
        return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        return jsonify({'success': False, 'error': f'运行时错误: {str(e)[:200]}'})


@ml_bp.route('/predict', methods=['POST'])
def api_predict():
    global _current_model
    try:
        data = request.get_json()
        if not data or 'values' not in data:
            return jsonify({'success': False, 'error': '请提供输入值'})
        result, error = predict_with_model(_current_model, data['values'])
        if error:
            return jsonify({'success': False, 'error': error})
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)[:200]})
