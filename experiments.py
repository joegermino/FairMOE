import DataLoader
from sklearn.model_selection import train_test_split
import time
from FairMOE import FairMOE
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import baselines.fairlearn_bl as fl
import baselines.xFAIR as xf
import baselines.zafar as z
import baselines.fairgbm_classifier as fgbm
import FairnessMetrics as fm
from lightgbm import LGBMClassifier
import json
import copy
from sklearn.model_selection import GridSearchCV
import argparse

def get_dataset(dataset_name):
    if dataset_name == 'dutch_census':
        X, y = DataLoader.get_dutch_census_data()
        protected_classes = [{'type': 'binary', 'name': 'sex_2'}]
    elif dataset_name == 'adult':
        X, y = DataLoader.get_adult_data()
        protected_classes = [{'type': 'continuous', 'name': 'age', 'privileged_range_min': 25, 'privileged_range_max': 60}, {'type': 'binary', 'name': 'sex__Male'}, {'type': 'categorical', 'dependent_columns': ['race__Asian-Pac-Islander', 'race__Black', 'race__Other', 'race__White'], 'privileged_classes': ['race__White'], 'name': 'race'}]
    elif dataset_name == 'german_credit':
        X, y = DataLoader.get_german_credit_data()
        protected_classes = [{'type': 'continuous', 'name': 'age_in_years', 'privileged_range_min': 25, 'privileged_range_max': 75}, {'type': 'binary', 'name': 'sex'}]
    elif dataset_name == 'bank_marketing':
        X, y = DataLoader.get_bank_marketing_data()
        # Married is majority class so treating as privileged
        protected_classes = [{'type': 'continuous', 'name': 'age', 'privileged_range_min': 25, 'privileged_range_max': 60}, {'type': 'categorical', 'dependent_columns': ['marital_married', 'marital_single'], 'privileged_classes': ['marital_married'], 'name': 'marital_status'}]
    elif dataset_name == 'credit_card_clients':
        X, y = DataLoader.get_credit_card_data()
        # Single is majority class and lowest default rate conditional probability so treating that as the privileged class
        # Education is included as a protected class in the survey but it's encoding is inconsistent with the dataset description so it's unclear how to apply here. Currently being ignored
        protected_classes = [{'type': 'binary', 'name': 'SEX'}, {'type': 'categorical', 'name': 'MARRIAGE', 'dependent_columns': ['MARRIAGE_1', 'MARRIAGE_2', 'MARRIAGE_3'], 'privileged_classes': ['MARRIAGE_2']}]
    elif dataset_name == 'oulad':
        X, y = DataLoader.get_oulad_data()
        protected_classes = [{'type': 'binary', 'name': 'gender'}]
    elif dataset_name == 'lawschool':
        X, y = DataLoader.get_lawschool_data()
        protected_classes = [{'type': 'binary', 'name': 'male'}, {'type': 'binary', 'name': 'race'}]
    else:
        raise ValueError(f"{dataset_name} dataset not found")
    return X, y, protected_classes

def eval_perf(preds, y_test) -> dict:
    accuracy = sum(preds == y_test) / y_test.shape[0]
    tp = preds[y_test==1].sum()
    fp = preds[y_test==0].sum()
    fn = preds[(y_test==1) & (preds==0)].shape[0]
    tn = preds[(y_test==0) & (preds==0)].shape[0]
    # precision = tp/(tp+fp) if tp+fp != 0 else np.nan
    recall = tp/(tp+fn) if tp+fn != 0 else np.nan
    specificity = tn/(tn+fp) if tn + fp != 0 else np.nan
    f1 = f1_score(y_test, preds)
    gmean = np.sqrt(specificity*recall)
    results = {}
    results['acc'] = accuracy
    results['f1'] = f1
    results['gmean'] = gmean
    return results

def eval_fair(preds, X_test, y_test, pcs) -> dict:
    results = {}
    X_test = X_test.copy()
    results['sp'] = {}
    results['eo']= {}
    for pc in pcs:
        if pc['type'] == 'binary':
            pc_col = pc['name']
        elif pc['type'] == 'continuous':
            pc_col = f'privileged_{pc["name"]}'
            X_test[pc_col] = X_test[pc['name']].apply(lambda x: int((x >= pc['privileged_range_min']) & (x <= pc['privileged_range_max']))) 
        elif pc['type'] == 'categorical':
            pc_col = f'privileged_{pc["name"]}'
            if all(item in list(X_test.columns) for item in pc['dependent_columns']):
                X_test[pc_col] = X_test[pc['privileged_classes']].sum(axis=1)
            else:
                X_test[pc_col] = X_test[pc['name']]
        results['sp'][pc['name']] = round(fm.statistical_parity(preds, X_test, pc_col), 3)
        results['eo'][pc['name']] = round(fm.equalized_odds(preds, X_test, y_test, pc_col), 3)
    return results

def eval_fm_overall(X_train, X_test, y_train, y_test, pcs, dataset, seed, label) -> dict:
    results = {}
    fairmoe = FairMOE(train_split=.5, non_interp_budget=1.0, verbose=False, seed=seed)
    fit_start = time.process_time()
    fairmoe.fit(X_train, y_train)
    fit_time = time.process_time() - fit_start
    
    fairmoe.set_non_interp_budget(1.0)
    fairmoe.set_use_counterfactual_fairness(True)
    predict_start = time.process_time()
    preds = fairmoe.predict(X_test, pcs)
    predict_time = time.process_time() - predict_start
    results['FairMOE_1.0'] = {}
    results['FairMOE_1.0']['time'] = fit_time + predict_time
    results['FairMOE_1.0'].update(eval_perf(preds, y_test))
    results['FairMOE_1.0'].update(eval_fair(preds, X_test, y_test, pcs))
    
    fairmoe.set_non_interp_budget(0.0)
    fairmoe.set_use_counterfactual_fairness(True)

    predict_start = time.process_time()
    preds = fairmoe.predict(X_test, pcs)
    predict_time = time.process_time() - predict_start
    results['FairMOE_0.0'] = {}
    results['FairMOE_0.0']['time'] = fit_time + predict_time
    results['FairMOE_0.0'].update(eval_perf(preds, y_test))
    results['FairMOE_0.0'].update(eval_fair(preds, X_test, y_test, pcs))
    
    fairmoe.set_non_interp_budget(0.0)
    fairmoe.set_use_counterfactual_fairness(False)

    predict_start = time.process_time()
    preds = fairmoe.predict(X_test, pcs)
    predict_time = time.process_time() - predict_start
    results['MOE_0.0'] = {}
    results['MOE_0.0']['time'] = fit_time + predict_time
    results['MOE_0.0'].update(eval_perf(preds, y_test))
    results['MOE_0.0'].update(eval_fair(preds, X_test, y_test, pcs))

    fairmoe.set_non_interp_budget(1.0)
    fairmoe.set_use_counterfactual_fairness(False)
    predict_start = time.process_time()
    preds = fairmoe.predict(X_test, pcs)
    predict_time = time.process_time() - predict_start
    results['MOE_1.0'] = {}
    results['MOE_1.0']['time'] = fit_time + predict_time
    results['MOE_1.0'].update(eval_perf(preds, y_test))
    results['MOE_1.0'].update(eval_fair(preds, X_test, y_test, pcs))

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    for expert in fairmoe.experts:
        mdl = copy.deepcopy(expert['model'])
        start_time = time.process_time()
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test_norm)
        end_time = time.process_time()
        results[expert['name']] = {}
        results[expert['name']]['time'] = end_time - start_time
        results[expert['name']].update(eval_perf(preds, y_test))
        results[expert['name']].update(eval_fair(preds, X_test, y_test, pcs))
    
    predict_start = time.process_time()
    preds = fairmoe.get_expert_predictions(X_test, pcs)
    mode_preds = preds.loc[:, [i['name'] for i in fairmoe.experts]].mode(axis=1)
    fair_mode_preds = preds.apply(lambda x: x[x['cf_fair_models']].mode()[0], axis=1)
    predict_time = time.process_time() - predict_start

    results['Mode'] = {}
    results['Mode'].update(eval_perf(np.ravel(mode_preds), y_test))
    results['Mode'].update(eval_fair(np.ravel(mode_preds), X_test, y_test, pcs))
    results['Mode']['time'] = fit_time + predict_time

    results['FairMode'] = {}
    results['FairMode'].update(eval_perf(fair_mode_preds, y_test))
    results['FairMode'].update(eval_fair(fair_mode_preds, X_test, y_test, pcs))
    results['FairMode']['time'] = fit_time + predict_time
    return results

def eval_fm_ensembles(X_train, X_test, y_train, y_test, pcs, dataset, seed, label) -> dict:
    results = {}
    fairmoe = FairMOE(train_split=.5, non_interp_budget=1.0, verbose=True, seed=seed)
    fairmoe.fit(X_train, y_train)
    preds = fairmoe.get_expert_predictions(X_test, pcs)
    mode_preds = preds.loc[:, [i['name'] for i in fairmoe.experts]].mode(axis=1)
    fair_mode_preds = preds.apply(lambda x: x[x['cf_fair_models']].mode()[0], axis=1)

    results['Mode'] = {}
    results['Mode'].update(eval_perf(np.ravel(mode_preds), y_test))
    results['Mode'].update(eval_fair(np.ravel(mode_preds), X_test, y_test, pcs))

    results['FairMode'] = {}
    results['FairMode'].update(eval_perf(fair_mode_preds, y_test))
    results['FairMode'].update(eval_fair(fair_mode_preds, X_test, y_test, pcs))

    return results

def eval_fm_ni_budget(X_train, X_test, y_train, y_test, pcs, dataset, seed, label) -> dict:
    results = {}
    fairmoe = FairMOE(train_split=.5, non_interp_budget=1.0, verbose=False, seed=seed)
    fairmoe.fit(X_train, y_train)
    for fairness_module in [True, False]:
        budget_results = {}
        fairmoe.set_use_counterfactual_fairness(fairness_module)
        for budget in np.linspace(0, 1.0, 21):
            fairmoe.set_non_interp_budget(round(budget, 2))
            preds = fairmoe.predict(X_test, pcs)
            
            budget_results[round(budget, 2)] = eval_perf(preds, y_test)

            replacement = {'Random_Forest': 'NI', 'XGB': 'NI', 'LGBM': 'NI' , 'Logistic_Regression': 'Int', 'Naive_Bayes': 'Int', 'Decision_Tree': 'Int', 'KNN': 'Int'}
            temp = fairmoe.expert_preds.copy()
            temp['selected_model'] = temp['selected_model'].replace(replacement)
            
            budget_results[round(budget, 2)]['Global NI Error Rate'] = sum(preds[temp['selected_model'] == 'NI'] != y_test[temp['selected_model'] == 'NI'])/len(y_test)
            budget_results[round(budget, 2)]['Predictions'] = list(preds)
            budget_results[round(budget, 2)]['Selected Models'] = list(fairmoe.expert_preds['selected_model'].copy())
            budget_results[round(budget, 2)]['Ground Truth'] = list(y_test)

        if fairness_module:
            results['FairMOE'] = budget_results
        else:
            results['MOE'] = budget_results
    return results

def eval_agarwal(X_train, X_test, y_train, y_test, pcs) -> dict:
    results = {}
    results['Agarwal'] = {}
    model = GridSearchCV(LGBMClassifier(), 
                         {'n_estimators': [10, 50, 100, 250], 
                          'learning_rate': [.001, .01, .1], 
                          'min_child_samples': [5, 10, 25]}, 
                        n_jobs=-1, 
                        scoring='roc_auc', 
                        cv=10)
    start_time = time.process_time()
    model.fit(X_train, y_train)
    preds = fl.fl_mitigator(X_train, y_train, X_test, pcs, model.best_estimator_)
    end_time = time.process_time()
    results['Agarwal']['time'] = end_time - start_time
    results['Agarwal'].update(eval_perf(preds, y_test))
    results['Agarwal'].update(eval_fair(preds, X_test, y_test, pcs))

    return results

def eval_hardt(X_train, X_test, y_train, y_test, pcs, seed) -> dict:
    results = {}
    results['Hardt'] = {}
    model = GridSearchCV(LGBMClassifier(), 
                         {'n_estimators': [10, 50, 100, 250], 
                          'learning_rate': [.001, .01, .1], 
                          'min_child_samples': [5, 10, 25]}, 
                        n_jobs=-1, 
                        scoring='roc_auc', 
                        cv=10)
    start_time = time.process_time()
    preds = fl.fl_postprocessor(X_train, y_train, X_test, pcs, model, seed)
    end_time = time.process_time()
    results['Hardt']['time'] = end_time - start_time
    results['Hardt'].update(eval_perf(preds, y_test))
    results['Hardt'].update(eval_fair(preds, X_test, y_test, pcs))

    return results

def eval_adversarial_debiasing(X_train, X_test, y_train, y_test, pcs, seed) -> dict:
    results = {}
    results['AdvDeb'] = {}
    start_time = time.process_time()
    preds = fl.fl_adversarial_debiasing(X_train, y_train, X_test, pcs, seed)
    end_time = time.process_time()
    results['AdvDeb']['time'] = end_time - start_time
    results['AdvDeb'].update(eval_perf(preds, y_test))
    results['AdvDeb'].update(eval_fair(preds, X_test, y_test, pcs))

    return results

def eval_xFAIR(X_train, X_test, y_train, y_test, pcs, seed) -> dict:
    results = {}
    results['xFAIR'] = {}
    pc_names = []
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    for pc in pcs:
        pc_names.append(pc['name'])
        if pc['type'] == 'continuous':
            X_train_norm.loc[:, pc['name']] = X_train_norm.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)
            X_test_norm.loc[:, pc['name']] = X_test_norm.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)
        if pc['type'] == 'categorical':
            cols_to_drop = []
            for col in pc['dependent_columns']:
                cols_to_drop.append(col)
            X_train_norm.loc[:, pc['name']] = X_train_norm[pc['privileged_classes']].any(axis=1).astype(int)
            X_train_norm = X_train_norm.drop(columns=cols_to_drop, axis=1)
            X_test_norm.loc[:, pc['name']] = X_test_norm[pc['privileged_classes']].any(axis=1).astype(int)
            X_test_norm = X_test_norm.drop(columns=cols_to_drop, axis=1)

    scaler = MinMaxScaler()
    X_train_norm = pd.DataFrame(scaler.fit_transform(X_train_norm), columns=X_train_norm.columns, index=X_train_norm.index)
    X_test_norm = pd.DataFrame(scaler.transform(X_test_norm), columns=X_test_norm.columns, index=X_test_norm.index)

    start_time = time.process_time()
    preds = xf.xFAIR(X_train_norm, y_train, X_test_norm, pc_names, seed)
    end_time = time.process_time()
    results['xFAIR']['time'] = end_time - start_time
    results['xFAIR'].update(eval_perf(preds, y_test))
    results['xFAIR'].update(eval_fair(preds, X_test, y_test, pcs))

    return results

def eval_zafar(X_train, X_test, y_train, y_test, pcs) -> dict:
    results = {}
    results['Zafar'] = {}
    pc_names = []
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    for pc in pcs:
        pc_names.append(pc['name'])
        if pc['type'] == 'continuous':
            X_train_norm.loc[:, pc['name']] = X_train_norm.apply(
                lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']),
                axis=1)
            X_test_norm.loc[:, pc['name']] = X_test_norm.apply(
                lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']),
                axis=1)
        if pc['type'] == 'categorical':
            X_train_norm.loc[:, pc['name']] = X_train_norm[pc['privileged_classes']].any(axis=1).astype(int)
            cols_to_drop = pc['dependent_columns']
            if pc['name'] in cols_to_drop:
                cols_to_drop.remove(pc['name'])
            X_train_norm.drop(columns=cols_to_drop, axis=1)
            X_train_norm.loc[:, pc['name']] = X_train_norm[pc['privileged_classes']].any(axis=1).astype(int)
            X_train_norm = X_train_norm.drop(columns=cols_to_drop, axis=1)
            X_test_norm.loc[:, pc['name']] = X_test_norm[pc['privileged_classes']].any(axis=1).astype(int)
            X_test_norm = X_test_norm.drop(columns=cols_to_drop, axis=1)
    y_train = y_train.astype(int).replace(0, -1)
    y_test = y_test.astype(int).replace(0, -1)

    scaler = MinMaxScaler()
    X_train_norm = pd.DataFrame(scaler.fit_transform(X_train_norm), columns=X_train_norm.columns, index=X_train_norm.index)
    X_test_norm = pd.DataFrame(scaler.transform(X_test_norm), columns=X_test_norm.columns, index=X_test_norm.index)

    start_time = time.process_time()
    preds = z.fairness_constraints_paper(X_train_norm, y_train, X_test_norm, y_test, pc_names)
    end_time = time.process_time()
    preds[preds==-1] = 0

    y_test = y_test.replace(-1, 0)
    results['Zafar']['time'] = end_time - start_time
    results['Zafar'].update(eval_perf(preds, y_test))
    results['Zafar'].update(eval_fair(preds, X_test, y_test, pcs))

    return results

def eval_fairgbm(X_train, X_test, y_train, y_test, pcs, seed):
    results = {}
    results['FairGBM'] = {}
    pc_names = []
    X_train = X_train.copy()
    X_test = X_test.copy()
    for pc in pcs:
        pc_names.append(pc['name'])
        if pc['type'] == 'continuous':
            X_train.loc[:, pc['name']] = X_train.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)
            X_test.loc[:, pc['name']] = X_test.apply(lambda x: int(pc['privileged_range_min'] <= x[pc['name']] <= pc['privileged_range_max']), axis=1)
        if pc['type'] == 'categorical':
            cols_to_drop = []
            for col in pc['dependent_columns']:
                cols_to_drop.append(col)
            X_train.loc[:, pc['name']] = X_train[pc['privileged_classes']].any(axis=1).astype(int)
            X_train = X_train.drop(columns=cols_to_drop, axis=1)
            X_test.loc[:, pc['name']] = X_test[pc['privileged_classes']].any(axis=1).astype(int)
            X_test = X_test.drop(columns=cols_to_drop, axis=1)
    
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    start_time = time.process_time()
    preds = fgbm.evaluate_fairgbm(X_train, X_test, y_train, pc_names, seed)
    end_time = time.process_time()
    results['FairGBM']['time'] = end_time - start_time
    results['FairGBM'].update(eval_perf(preds, y_test))
    results['FairGBM'].update(eval_fair(preds, X_test, y_test, pcs))

    return results

def main(label):
    datasets = ['dutch_census', 'adult', 'german_credit', 'credit_card_clients', 'bank_marketing', 'oulad', 'lawschool']
    results = dict()
    fm_budget_dict = dict()
    for dataset in datasets:
        results[dataset] = {}
        X, y, pcs = get_dataset(dataset)
        np.random.seed(int(time.time()))
        seed = np.random.randint(-2**31,2**31-1)+2**31
        results[dataset]['seed'] = seed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        
        fm_dict = eval_fm_overall(X_train, X_test, y_train, y_test, pcs, dataset, seed, label)
        agarwal_dict = eval_agarwal(X_train, X_test, y_train, y_test, pcs)
        xfair_dict = eval_xFAIR(X_train, X_test, y_train, y_test, pcs, seed)
        zafar_dict = eval_zafar(X_train, X_test, y_train, y_test, pcs)
        hardt_dict = eval_hardt(X_train, X_test, y_train, y_test, pcs, seed)

        results[dataset].update(fm_dict)
        results[dataset].update(agarwal_dict)
        results[dataset].update(xfair_dict)
        results[dataset].update(zafar_dict)
        results[dataset].update(hardt_dict)

        fm_budget_dict[dataset] = eval_fm_ni_budget(X_train, X_test, y_train, y_test, pcs, dataset, seed, label)

    with open(f"results/results{label}.json", 'w') as fp:
        json.dump(results, fp)

    with open(f"results/budget_results{label}.json", 'w') as fp:
        json.dump(fm_budget_dict, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', default='')
    args = parser.parse_args()
    main(args.label)
