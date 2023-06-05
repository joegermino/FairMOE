from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.adversarial import AdversarialFairnessClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def fl_mitigator(train_X, train_y, test_X, protected_classes, model):
    train_A = pd.DataFrame()
    test_A = pd.DataFrame()
    for pc in protected_classes:
        if pc['type'] == 'binary':
            train_A[pc['name']] = train_X[pc['name']].astype(int)
            test_A[pc['name']] = test_X[pc['name']].astype(int)
        elif pc['type'] == 'categorical':
            train_A[pc['name']] = train_X[pc['privileged_classes']].any(axis=1).astype(int)
            test_A[pc['name']] = test_X[pc['privileged_classes']].any(axis=1).astype(int)
        elif pc['type'] == 'continuous':
            train_A[pc['name']] = train_X[pc['name']]
            test_A[pc['name']] = test_X[pc['name']]
            train_A.loc[:, pc['name']] = train_A.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)
            test_A.loc[:, pc['name']] = test_A.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)

    scaler = MinMaxScaler()
    train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=train_X.columns, index=train_X.index)
    test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns, index=test_X.index)

    constraint = EqualizedOdds()
    mitigator = ExponentiatedGradient(model, constraint)

    mitigator.fit(train_X, train_y, sensitive_features=train_A)
    mitigation_preds = mitigator.predict(test_X)
    return mitigation_preds


def fl_postprocessor(X_train, y_train, X_test, protected_classes, model, seed):
    A_train = pd.DataFrame()
    A_test = pd.DataFrame()
    for pc in protected_classes:
        if pc['type'] == 'binary':
            A_train[pc['name']] = X_train[pc['name']].astype(int)
            A_test[pc['name']] = X_test[pc['name']].astype(int)
        elif pc['type'] == 'categorical':
            A_train[pc['name']] = X_train[pc['privileged_classes']].any(axis=1).astype(int)
            A_test[pc['name']] = X_test[pc['privileged_classes']].any(axis=1).astype(int)
        elif pc['type'] == 'continuous':
            A_train[pc['name']] = X_train[pc['name']]
            A_test[pc['name']] = X_test[pc['name']]
            A_train.loc[:, pc['name']] = A_train.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)
            A_test.loc[:, pc['name']] = A_test.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    postproc_est = ThresholdOptimizer(estimator=model, constraints="equalized_odds", prefit=False, predict_method='predict')

    if (y_train == 1).sum() <= (y_train == 0).sum():
        balanced_idx1 = X_train[y_train == 1].index
        pp_train_idx = balanced_idx1.union(y_train[y_train == 0].sample(n=balanced_idx1.size).index)
    else:
        balanced_idx1 = X_train[y_train == 0].index
        pp_train_idx = balanced_idx1.union(y_train[y_train == 1].sample(n=balanced_idx1.size).index)
    X_train_balanced = X_train.loc[pp_train_idx, :]
    y_train_balanced = y_train.loc[pp_train_idx]
    A_train_balanced = A_train.loc[pp_train_idx]

    postproc_est.fit(X_train_balanced, y_train_balanced, sensitive_features=A_train_balanced)
    postproc_preds = postproc_est.predict(X_test, sensitive_features=A_test, random_state=seed)
    return postproc_preds

def fl_adversarial_debiasing(X_train, y_train, X_test, protected_classes, seed):
    A_train = pd.DataFrame()
    A_test = pd.DataFrame()
    for pc in protected_classes:
        if pc['type'] == 'binary':
            A_train[pc['name']] = X_train[pc['name']].astype(int)
            A_test[pc['name']] = X_test[pc['name']].astype(int)
        elif pc['type'] == 'categorical':
            A_train[pc['name']] = X_train[pc['privileged_classes']].any(axis=1).astype(int)
            A_test[pc['name']] = X_test[pc['privileged_classes']].any(axis=1).astype(int)
        elif pc['type'] == 'continuous':
            A_train[pc['name']] = X_train[pc['name']]
            A_test[pc['name']] = X_test[pc['name']]
            A_train.loc[:, pc['name']] = A_train.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)
            A_test.loc[:, pc['name']] = A_test.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)
    
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    if 'age' in A_train.columns:
        A_train = A_train['age']

    predictor_model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)])
    adversary_model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(1)])
    advdeb = AdversarialFairnessClassifier(predictor_model=predictor_model, adversary_model=adversary_model, random_state=seed)
    advdeb.fit(X_train, y_train, sensitive_features=A_train)
    preds = advdeb.predict(X_test)
    preds.index = X_test.index
    return preds