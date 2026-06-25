import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


class MLInputError(Exception):
    pass


def _prepare_data(df, feature_cols, target_col, test_size=0.3, random_state=42):
    # 自动剔除目标列（防止数据泄露）
    feature_cols = [c for c in feature_cols if c != target_col]
    if not feature_cols:
        raise MLInputError('特征列不能为空（目标列已自动排除）')
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.select_dtypes(include=[np.number]).fillna(X.mean())
    is_clf = _is_classification(df[target_col])
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20 and is_clf:
        raise MLInputError(
            f'目标列为连续数值，分类模型需要离散类别。请选择类别型目标列或使用 K-Means 聚类。')
    if y.dtype == 'object' or str(y.dtype) == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        target_names = list(le.classes_)
    else:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        target_names = sorted(df[target_col].unique().tolist())
        target_names = [str(t) for t in target_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, list(X.columns), target_names, is_clf


def _is_classification(y):
    if pd.api.types.is_numeric_dtype(y):
        return y.nunique() <= 20
    return True


def _build_clf_result(model, X_test, y_test, y_pred, features, target_names, model_name):
    return {
        'task_type': 'classification', 'model_name': model_name,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'target_names': target_names,
        'classification_report': classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0),
        'features': features, 'n_features': len(features),
        'n_samples': len(y_test), 'n_classes': len(target_names),
        'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()
    }


def _build_reg_result(model, X_test, y_test, y_pred, features, target_names, model_name):
    mse = mean_squared_error(y_test, y_pred)
    return {
        'task_type': 'regression', 'model_name': model_name,
        'mse': round(mse, 4), 'rmse': round(np.sqrt(mse), 4),
        'mae': round(mean_absolute_error(y_test, y_pred), 4),
        'r2': round(r2_score(y_test, y_pred), 4),
        'features': features, 'n_features': len(features),
        'n_samples': len(y_test),
        'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()
    }


# ===== K-Means 聚类 =====
def kmeans_clustering(df, feature_cols, n_clusters=3, random_state=42):
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.select_dtypes(include=[np.number]).fillna(X.mean())
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X).tolist()
    counts = {int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()}
    return {
        'task_type': 'clustering', 'model_name': 'K-Means',
        'labels': labels.tolist(), 'centers': model.cluster_centers_.tolist(),
        'silhouette_score': round(score, 4), 'cluster_counts': counts,
        'n_clusters': n_clusters, 'inertia': round(model.inertia_, 2),
        'features': feature_cols, 'coords': coords, 'n_samples': len(X)
    }


# ===== 朴素贝叶斯 =====
def naive_bayes(df, feature_cols, target_col, test_size=0.3):
    X_train, X_test, y_train, y_test, features, tn, _ = _prepare_data(df, feature_cols, target_col, test_size)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _build_clf_result(model, X_test, y_test, y_pred, features, tn, '朴素贝叶斯'), (model, None, features, tn, True)


# ===== 神经网络 =====
def neural_network(df, feature_cols, target_col, test_size=0.3, hidden_layers='100,50', max_iter=500, activation='relu'):
    X_train, X_test, y_train, y_test, features, tn, is_clf = _prepare_data(df, feature_cols, target_col, test_size)
    layers = tuple(int(x.strip()) for x in hidden_layers.split(','))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = (MLPClassifier if is_clf else MLPRegressor)(
        hidden_layer_sizes=layers, max_iter=max_iter, activation=activation,
        random_state=42, early_stopping=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    builder = _build_clf_result if is_clf else _build_reg_result
    result = builder(model, X_test, y_test, y_pred, features, tn, '神经网络')
    result['params'] = {'hidden_layers': hidden_layers, 'max_iter': max_iter, 'activation': activation}
    result['loss'] = round(model.loss_curve_[-1], 4) if hasattr(model, 'loss_curve_') and model.loss_curve_ else 0
    return result, (model, scaler, features, tn, is_clf)


# ===== 支持向量机 =====
def svm_classify(df, feature_cols, target_col, test_size=0.3, kernel='rbf', C=1.0, gamma='scale'):
    X_train, X_test, y_train, y_test, features, tn, is_clf = _prepare_data(df, feature_cols, target_col, test_size)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42, probability=True) if is_clf else SVR(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if not is_clf:
        y_pred = np.round(y_pred).astype(int)
    builder = _build_clf_result if is_clf else _build_reg_result
    result = builder(model, X_test, y_test, y_pred, features, tn, 'SVM' if is_clf else 'SVR')
    result['params'] = {'kernel': kernel, 'C': C, 'gamma': gamma}
    if is_clf:
        result['n_support_vectors'] = int(sum(model.n_support_))
    return result, (model, scaler, features, tn, is_clf)


# ===== 决策树 =====
def decision_tree(df, feature_cols, target_col, test_size=0.3, max_depth=None, min_samples_split=2, criterion='gini'):
    X_train, X_test, y_train, y_test, features, tn, is_clf = _prepare_data(df, feature_cols, target_col, test_size)
    model = (DecisionTreeClassifier if is_clf else DecisionTreeRegressor)(
        max_depth=max_depth, min_samples_split=min_samples_split,
        random_state=42, **(dict(criterion=criterion) if is_clf else {}))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if not is_clf:
        y_pred = np.round(y_pred).astype(int)
    builder = _build_clf_result if is_clf else _build_reg_result
    result = builder(model, X_test, y_test, y_pred, features, tn, '决策树')
    result['params'] = {'max_depth': model.tree_.max_depth, 'min_samples_split': min_samples_split}
    result['tree_text'] = export_text(model, feature_names=features)[:2000]
    imp = {features[i]: round(model.feature_importances_[i], 4) for i in range(len(features))}
    result['feature_importance'] = dict(sorted(imp.items(), key=lambda x: -x[1]))
    return result, (model, None, features, tn, is_clf)


# ===== 特征选择 =====
def feature_selection(df, feature_cols, target_col, k=5, method='f_classif'):
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))
    X = X.select_dtypes(include=[np.number]).fillna(X.mean())
    sf = {'f_classif': f_classif, 'chi2': chi2, 'mutual_info': mutual_info_classif}.get(method, f_classif)
    if method == 'chi2':
        X = X - X.min() + 1e-10
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=sf, k=k)
    selector.fit(X, y)
    results = [{'feature': col, 'score': round(float(s), 4), 'selected': bool(m)}
               for col, s, m in zip(X.columns, selector.scores_, selector.get_support())]
    results.sort(key=lambda x: -x['score'])
    return {'task_type': 'feature_selection', 'method': method, 'k': k,
            'results': results, 'selected_features': [r['feature'] for r in results if r['selected']]}, None


# ===== K近邻 =====
def knn_classify(df, feature_cols, target_col, test_size=0.3, n_neighbors=5, weights='distance', metric='euclidean'):
    X_train, X_test, y_train, y_test, features, tn, is_clf = _prepare_data(df, feature_cols, target_col, test_size)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = (KNeighborsClassifier if is_clf else KNeighborsRegressor)(
        n_neighbors=n_neighbors, weights=weights, metric=metric)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if not is_clf:
        y_pred = np.round(y_pred).astype(int)
    builder = _build_clf_result if is_clf else _build_reg_result
    result = builder(model, X_test, y_test, y_pred, features, tn, 'K近邻')
    result['params'] = {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric}
    return result, (model, scaler, features, tn, is_clf)


# ===== Ridge回归 =====
def ridge_regression(df, feature_cols, target_col, test_size=0.3, alpha=1.0):
    X_train, X_test, y_train, y_test, features, tn, is_clf = _prepare_data(df, feature_cols, target_col, test_size)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = _build_reg_result(model, X_test, y_test, y_pred, features, tn, 'Ridge回归')
    result['params'] = {'alpha': alpha}
    result['coefficients'] = dict(zip(features, [round(c, 4) for c in model.coef_]))
    return result, (model, scaler, features, tn, is_clf)


# ===== 预测 =====
def predict_with_model(model_data, input_values):
    if not model_data:
        return None, '没有已训练的模型，请先运行算法'
    model, scaler, features, target_names, is_clf = model_data
    df_input = pd.DataFrame([input_values])
    df_input = df_input[features]
    X = scaler.transform(df_input) if scaler else df_input.values
    pred = model.predict(X)
    pred_label = int(pred[0])
    pred_name = target_names[pred_label] if is_clf and pred_label < len(target_names) else str(pred_label)
    return {'label': pred_label, 'name': pred_name}, None
