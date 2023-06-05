import numpy as np
import baselines.zafar_loss_funcs as lf
import baselines.zafar_utils as ut
import baselines.zafar_funcs_disp_mist as fdm


def fairness_constraints_paper(train_X, train_y, test_X, test_y, pcs):
    # http://proceedings.mlr.press/v54/zafar17a/zafar17a.pdf
    train_X_control = train_X[pcs]
    train_X = train_X.drop(pcs, axis=1)
    test_X_control = test_X[pcs]
    test_X = test_X.drop(pcs, axis=1)
    train_X.loc[:, 'intercept'] = 1
    test_X.loc[:, 'intercept'] = 1

    train_X_control = train_X_control.reset_index(drop=True)
    train_X = train_X.reset_index(drop=True)
    test_X_control = test_X_control.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    apply_fairness_constraints = None
    apply_accuracy_constraint = None
    sep_constraint = None

    loss_function = lf._logistic_loss
    sensitive_attrs = pcs
    sensitive_attrs_to_cov_thresh = {}
    gamma = None

    def train_test_classifier():
        w = ut.train_model(train_X, train_y, train_X_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
        distances_boundary_test = (np.dot(test_X, w)).tolist()
        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, train_X,
                                                                                                    train_y, test_X,
                                                                                                    test_y, None, None)
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, test_X_control, pcs)

        return all_class_labels_assigned_test

    apply_fairness_constraints = 0  # flag for fairness constraint is set to 0 to apply the accuracy constraint
    apply_accuracy_constraint = 1  # want to optimize accuracy subject to fairness constraints
    sep_constraint = 1  # set the separate constraint flag to one, no misclassifications for certain points
    gamma = 1000.0
    preds = train_test_classifier()
    return preds


def fairness_beyond_di_paper(train_X, train_y, test_X, test_y, pcs):
    # https://arxiv.org/pdf/1610.08452.pdf
    train_X_control = train_X[pcs]
    train_X = train_X.drop(pcs, axis=1)
    test_X_control = test_X[pcs]
    test_X = test_X.drop(pcs, axis=1)
    train_X.loc[:, 'intercept'] = 1
    test_X.loc[:, 'intercept'] = 1

    train_X_control = train_X_control.reset_index(drop=True)
    train_X = train_X.reset_index(drop=True)
    test_X_control = test_X_control.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    train_X_control = train_X_control.astype(int)
    test_X_control = test_X_control.astype(int)

    cons_params = None  # constraint parameters, will use them later
    loss_function = "logreg"  # perform the experiments with logistic regression
    EPS = 1e-6

    def train_test_classifier():
        w = fdm.train_model_disp_mist(train_X, train_y, train_X_control, loss_function, EPS, cons_params)

        distances_boundary_test = fdm.get_distance_boundary(w, test_X, test_X_control[pcs])
        preds = np.sign(distances_boundary_test)

        return preds

    cons_type = 1  # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
    tau = 5.0
    mu = 1.2
    sensitive_attrs_to_cov_thresh = {"race": {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}}  # zero covariance threshold, means try to get the fairest solution
    cons_params = {"cons_type": cons_type,
                   "tau": tau,
                   "mu": mu,
                   "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

    preds = train_test_classifier()
    return preds
