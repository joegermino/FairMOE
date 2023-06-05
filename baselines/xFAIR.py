from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from imblearn.over_sampling import SMOTE
import copy


def xFAIR(X_train, y_train, X_test, pcs, seed, smote1=True, thresh=.5):  # Also called FairMask
    base_clf = RandomForestClassifier(random_state=seed, n_jobs=-1)  # This is from the code on github
    base2 = DecisionTreeRegressor(random_state=seed)

    final_classification_model = copy.deepcopy(base_clf)
    final_classification_model.fit(X_train, y_train)

    reduced = list(X_train.columns)
    for col in pcs:
        reduced.remove(col)
    extrapolation_clfs = []
    for pc in pcs:
        X_reduced, y_reduced = X_train.loc[:, reduced], X_train[pc]

        clf1 = copy.deepcopy(base2)
        if smote1:
            sm = SMOTE()
            X_trains, y_trains = sm.fit_resample(X_reduced, y_reduced)
            if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
                clf1.fit(X_trains, y_trains)
            else:
                clf = copy.deepcopy(base_clf)
                clf.fit(X_trains, y_trains)
                y_proba = clf.predict_proba(X_trains)
                y_proba = [each[1] for each in y_proba]
                clf1.fit(X_trains, y_proba)
        else:
            if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
                clf1.fit(X_reduced, y_reduced)
            else:
                clf = copy.deepcopy(base_clf)
                clf.fit(X_reduced, y_reduced)
                y_proba = clf.predict_proba(X_reduced)
                y_proba = [each[1] for each in y_proba]
                clf1.fit(X_reduced, y_proba)
        extrapolation_clfs.append(clf1)

    X_test_reduced = X_test.loc[:, reduced]
    for i in range(len(extrapolation_clfs)):
        protected_pred = extrapolation_clfs[i].predict(X_test_reduced)
        if isinstance(extrapolation_clfs[i], DecisionTreeRegressor) or isinstance(extrapolation_clfs[i], LinearRegression):
            protected_pred = reg2clf(protected_pred, threshold=thresh)
        X_test.loc[:, pcs[i]] = protected_pred

    y_pred = final_classification_model.predict(X_test)

    return y_pred

def reg2clf(protected_pred, threshold=.5):
    out = []
    for each in protected_pred:
        if each >= threshold:
            out.append(1)
        else: out.append(0)
    return out