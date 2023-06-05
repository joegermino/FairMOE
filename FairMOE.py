import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from lightgbm import LGBMClassifier
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from itertools import combinations
from FairnessMetrics import statistical_parity
import time


class FairMOE:
    def __init__(self, train_split=.5, non_interp_budget=1.0, verbose=True, use_counterfactual_fairness=True, seed:int=None):
        '''
        train_split (float): percentage of training data to be used to train experts. The remaining train data will be used for the meta-learners (default 0.5) 
        non_interp_budget (float): maximum percentage of predictions to be made by non-interpretable experts (default 1.0)
        verbose (bool): indicates whether to run in verbose mode (default True)
        use_counterfactual_fairness (bool): indicates whether to include the Counterfactual Fairness Module (default True)
        seed (int): set a seed for reproducibility (default None)
        '''
        self.train_split = train_split
        self.non_interp_budget = non_interp_budget
        self.verbose = verbose
        self.use_counterfactual_fairness = use_counterfactual_fairness
        if seed is None:
            np.random.seed(int(time.time()))
            self.seed = np.random.randint(-2**31,2**31-1)+2**31
        else:
            self.seed = seed

        self.experts = []
        self.experts.append({'name': 'Logistic_Regression', 'model': LogisticRegression(solver='liblinear', random_state=self.seed),'interpretable': True})
        self.experts.append({'name': 'Naive_Bayes', 'model': BernoulliNB(),'interpretable': True})
        self.experts.append({'name': 'Decision_Tree', 'model': GridSearchCV(DecisionTreeClassifier(random_state=self.seed), {'max_depth': [3, 5, 10, 15], 'min_samples_leaf': [5, 10, 25]}, n_jobs=-1, scoring='roc_auc', cv=10), 'interpretable': True})
        self.experts.append({'name': 'KNN', 'model': GridSearchCV(KNeighborsClassifier(weights='distance'), {'n_neighbors': range(5, 56, 4)}, n_jobs=-1, scoring='roc_auc', cv=10), 'interpretable': True})
        self.experts.append({'name': 'Random_Forest', 'model': GridSearchCV(RandomForestClassifier(random_state=self.seed), {'n_estimators': [10, 50, 100, 250], 'min_samples_leaf': [5, 10, 25]}, n_jobs=-1, scoring='roc_auc', cv=10), 'interpretable': False})
        self.experts.append({'name': 'LGBM', 'model': GridSearchCV(LGBMClassifier(random_state=self.seed), {'n_estimators': [10, 50, 100, 250], 'learning_rate': [.001, .01, .1], 'min_child_samples': [5, 10, 25]}, n_jobs=-1, scoring='roc_auc', cv=10), 'interpretable': False})
        self.experts.append({'name': 'XGB', 'model': GridSearchCV(xgb.XGBClassifier(random_state=self.seed), {'n_estimators': [10, 50, 100, 250], 'learning_rate': [.001, .01, .1], 'max_depth': [3, 5, 10]}, n_jobs=-1, scoring='roc_auc', cv=10), 'interpretable': False})

        self.combos = None
        self.counterfactuals = None  

    def set_non_interp_budget(self, budget: float):
        self.non_interp_budget = budget

    def set_use_counterfactual_fairness(self, set: bool):
        self.use_counterfactual_fairness = set

    def set_use_group_fairness(self, set: bool):
        self.use_group_fairness = set

    def get_experts(self):
        return self.experts
    
    def set_experts(self, experts: list):
        for expert in experts:
            if type(expert) != dict:
                raise TypeError(f"{expert} must be a dict containing name, model, and interpretable")
            if 'name' not in expert:
                raise KeyError(f"{expert} must have a name (str)")
            if 'model' not in expert:
                raise KeyError(f"{expert} must have a model")
            if 'interpretable' not in expert:
                raise KeyError(f"{expert} must have an interpretable flag (bool)")
        self.experts = experts
            
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        '''
        Fit the experts and performance meta-learners

        X (pd.DataFrame): train data
        y (pd.DataFrame): ground truth values for X
        '''
        # Save train data so we can calculate the privileged classes later
        self._train_X = X
        # Normalize data - handling this internally to make it easier to identify privileged ranges when generating counterfactuals
        self.scaler = MinMaxScaler()
        cols = X.columns
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=cols)
        # Split train data in 2
        X1 = X.iloc[:int(self.train_split*X.shape[0]), :]
        y1 = y.iloc[:int(self.train_split*y.shape[0]),]
        X2 = X.iloc[int(self.train_split*X.shape[0]):, :]
        y2 = y.iloc[int(self.train_split*y.shape[0]):]
        
        self._fit_experts(X1, y1)
        self._fit_pmls(X2, y2)

    def predict(self, X: pd.DataFrame, protected_classes: dict):
        '''
        Predict data. Must be called after model has been fit

        X (pd.DataFrame): test data to predict
        protected_classes (dict): dictionary of protected attributes, types, and privileged ranges
        '''
        cols = X.columns
        X_norm = pd.DataFrame(self.scaler.transform(X), columns=cols, index=X.index)

        expert_proba_preds = self._proba_predict_experts(X_norm)
        expert_preds = expert_proba_preds.round().astype(int)
        meta_preds = self._predict_metas(X_norm, expert_proba_preds)

        if self.use_counterfactual_fairness:
            counterfactuals = self._create_counterfactuals(X, protected_classes)
            counterfactuals_norm = pd.DataFrame(self.scaler.transform(counterfactuals), columns=counterfactuals.columns, index=counterfactuals.index)
            counterfactual_preds = self._predict_counterfactuals(counterfactuals_norm)
            cf_fair_models = self._cf_fair_models(expert_preds, counterfactual_preds)
            self.counterfactual_preds = counterfactual_preds
            self.counterfactuals = counterfactuals
        else:
            counterfactual_preds = None
            cf_fair_models = pd.DataFrame(index=X.index)
            experts = [i['name'] for i in self.experts]
            cf_fair_models['cf_fair'] = [experts]*X.shape[0]
        # if self.use_group_fairness:
        #     if self.group_fair_method == 'gap':
        #         group_fair_models = self._largest_gap_group_fair_models(X, expert_preds, protected_classes)
        #     elif self.group_fair_method == 'pos_neg':
        #         group_fair_models = self._pos_neg_group_fair_models(X, expert_preds, protected_classes)
        # else:
        #     group_fair_models = pd.DataFrame(index=X.index)
        #     group_fair_models['group_fair'] = [list(counterfactual_preds.columns)]*X.shape[0]
        #     group_fair_models = pd.Series(group_fair_models['group_fair'])      
        self.cf_fair_models = cf_fair_models
        # self.group_fair_models = group_fair_models
        
        preds = self._assignment_module(expert_preds, meta_preds, cf_fair_models, counterfactual_preds) #, group_fair_models)
        return preds
    
    def get_expert_predictions(self, X: pd.DataFrame, protected_classes: dict) -> pd.DataFrame:
        '''
        Return DataFrame of all experts' predictions for X

        X (pd.DataFrame): test data to predict
        protected_classes (dict): dictionary of protected attributes, types, and privileged ranges
        '''
        cols = X.columns
        X_norm = pd.DataFrame(self.scaler.transform(X), columns=cols, index=X.index)
        expert_preds = self._predict_experts(X_norm)
        
        counterfactuals = self._create_counterfactuals(X, protected_classes)
        counterfactuals_norm = pd.DataFrame(self.scaler.transform(counterfactuals), columns=counterfactuals.columns, index=counterfactuals.index)
        counterfactual_preds = self._predict_counterfactuals(counterfactuals_norm)
        expert_preds['cf_fair_models'] = self._cf_fair_models(expert_preds, counterfactual_preds)

        return expert_preds

    def _fit_experts(self, X: pd.DataFrame, y: pd.DataFrame):
        if self.verbose:
            print("Training Experts")
        for expert in tqdm(self.experts, disable=not self.verbose):
            expert['model'].fit(X, y)

    def _fit_pmls(self, X: pd.DataFrame, y: pd.DataFrame):
        self.performance_meta_learners = []
        if self.verbose:
            print("Training Performance Meta Learners")
        for expert in tqdm(self.experts, disable=not self.verbose):
            # Include expert proba prediction in X
            proba_preds = expert['model'].predict_proba(X)[:,1]
            new_X = X.copy()
            new_X['expert_pred'] = proba_preds
            
            # Train on if the prediction was correct or not
            preds = proba_preds.round().astype(int)
            correct = np.array(preds == y, dtype=int)

            # Perform nested cross-validation to train/select meta learner for each expert
            potential_metas = []
            potential_metas.append(LogisticRegression(solver='liblinear', random_state=self.seed))
            potential_metas.append(BernoulliNB())
            potential_metas.append(GridSearchCV(DecisionTreeClassifier(random_state=self.seed), {'max_depth': [3, 5, 10, 15], 'min_samples_leaf': [5, 10, 25]}, n_jobs=-1, scoring='accuracy', cv=10))
            potential_metas.append(GridSearchCV(KNeighborsClassifier(weights='distance'), {'n_neighbors': range(5, 56, 4)}, n_jobs=-1, scoring='accuracy', cv=10))
            cross_val_scores = []
            for meta in potential_metas:
                cv = cross_val_score(meta, new_X, correct)
                cross_val_scores.append(cv.mean())
            pml = potential_metas[np.argmax(cross_val_scores)]
            pml.fit(new_X, correct)
            self.performance_meta_learners.append(pml)

    def _proba_predict_experts(self, X: pd.DataFrame):
        if self.verbose:
            print("Predicting Experts")
        expert_proba_preds = pd.DataFrame()
        for expert in tqdm(self.experts, disable=not self.verbose):
            proba_preds = expert['model'].predict_proba(X)
            expert_proba_preds[expert['name']] = np.take(proba_preds, 1, 1)
        expert_proba_preds.index = X.index
        return expert_proba_preds
    
    def _predict_experts(self, X: pd.DataFrame):
        if self.verbose:
            print("Predicting Experts")
        expert__preds = pd.DataFrame()
        for expert in tqdm(self.experts, disable=not self.verbose):
            expert__preds[expert['name']] = expert['model'].predict(X).astype(int)
        expert__preds.index = X.index
        return expert__preds

    def _predict_metas(self, X: pd.DataFrame, proba_preds: pd.DataFrame):
        if self.verbose:
            print("Predicting Metas")
        meta_preds = pd.DataFrame()
        for i in tqdm(range(len(self.performance_meta_learners)), disable=not self.verbose):
            new_X = X.copy()
            new_X['expert_pred'] = proba_preds[self.experts[i]['name']]
            meta_proba_preds = self.performance_meta_learners[i].predict_proba(new_X)
            meta_preds[self.experts[i]['name']] = np.take(meta_proba_preds, 1, 1)
        meta_preds.index = X.index
        return meta_preds

    def _create_counterfactuals(self, X: pd.DataFrame, protected_classes: dict):
        counterfactuals = pd.concat([X]*(2**len(protected_classes)-1))
        if self.verbose:
            print("Creating Counterfactuals")
        train_dists = self._calculate_train_dists(protected_classes)
        random.seed(self.seed)
        for i in tqdm(X.index, disable=not self.verbose):
            new_rows = self._generate_combos(X.loc[i,:], protected_classes, train_dists)
            for col in new_rows:
                if len(new_rows[col]) == 1:
                    counterfactuals.loc[i, col] = new_rows[col][0]
                else:
                    counterfactuals.loc[i, col] = new_rows[col]
        return counterfactuals

    def _calculate_train_dists(self, protected_classes: dict):
        '''
        Calculate distributions for all continuous/categorical features within the orignal train data. Will be used when generating counterfactuals

        protected_classes (dict): dictionary of protected attributes, types, and privileged ranges
        '''
        train_distributions = {}
        for pc in protected_classes:
            if pc['type'] == 'continuous':
                train_distributions[pc['name']] = list(self._train_X[pc['name']].copy())
            elif pc['type'] == 'categorical':
                dat = self._train_X.copy()
                dependent_columns = [col for col in pc['dependent_columns']]
                probs = []
                for col in dependent_columns:
                    probs.append(dat[dat[col]==1].shape[0])
                probs.append(dat.loc[(dat[dependent_columns]==0).all(axis=1),:].shape[0])
                dependent_columns.append('All Zeroes')
                train_distributions[pc['name']] = (dependent_columns, probs)
        return train_distributions

    def _generate_combos(self, original_row, protected_classes, train_dists):
        '''
        Generate permutations of privileged/unprivileged attributes for a given instance in the test data

        original_row (pd.DataFrame): instance needing counterfactuals
        protected_classes (dict): dictionary of protected attributes, types, and privileged ranges
        train_dists (dict): dictionary of distributions of original training data for continuous and categorical features
        '''
        if self.combos is None:
            self.combos = []
            for i in range(len(protected_classes)+1):
                self.combos.extend(list(combinations(range(len(protected_classes)), i)))
        new_rows = {}

        combos = self.combos.copy()
        # Find combination in original data and remove from combos
        orig_combo = []

        for i in range(len(protected_classes)):
            if protected_classes[i]['type'] == 'binary':
                if original_row[protected_classes[i]['name']] == 1:
                    orig_combo.append(i)
            elif protected_classes[i]['type'] == 'continuous':
                if protected_classes[i]['privileged_range_min'] <= original_row[protected_classes[i]['name']] <= protected_classes[i]['privileged_range_max']:
                    orig_combo.append(i)
            elif protected_classes[i]['type'] == 'categorical':
                for col in protected_classes[i]['privileged_classes']:
                    if original_row[col] == 1:
                        orig_combo.append(i)
                        break
            else:
                raise ValueError(f"Protected class type: {protected_classes[i]['type']} not in ['binary', 'continuous', 'categorical']")
        combos.remove(tuple(orig_combo))

        # Create other counterfactuals
        for combo in combos:
            for i in range(len(protected_classes)):
                if protected_classes[i]['type'] == 'binary':
                    if protected_classes[i]['name'] not in new_rows:
                        new_rows[protected_classes[i]['name']] = []
                    if i in combo: # Use privileged attribute
                        new_rows[protected_classes[i]['name']].append(1)
                    else: # Use non-privileged attribute
                        new_rows[protected_classes[i]['name']].append(0)
                elif protected_classes[i]['type'] == 'continuous':
                    # Pick from selection mimicking distribution of entire training data
                    if protected_classes[i]['name'] not in new_rows:
                        new_rows[protected_classes[i]['name']] = []
                    val = random.choice(train_dists[protected_classes[i]['name']])
                    if i in combo:
                        while val < protected_classes[i]['privileged_range_min'] or val > protected_classes[i]['privileged_range_max']:
                            val = random.choice(train_dists[protected_classes[i]['name']])
                        new_rows[protected_classes[i]['name']].append(val)
                    else:
                        while protected_classes[i]['privileged_range_min'] <= val <= protected_classes[i]['privileged_range_max']:
                            val = random.choice(train_dists[protected_classes[i]['name']])
                        new_rows[protected_classes[i]['name']].append(val)
                elif protected_classes[i]['type'] == 'categorical':
                    for col in protected_classes[i]['dependent_columns']:
                        if col not in new_rows:
                            new_rows[col] = []
                    val = random.choices(train_dists[protected_classes[i]['name']][0], train_dists[protected_classes[i]['name']][1])[0]
                    if i in combo:
                        # Pick from selection mimicking distribution of entire training data
                        while val not in protected_classes[i]['privileged_classes']:
                            val = random.choices(train_dists[protected_classes[i]['name']][0], train_dists[protected_classes[i]['name']][1])[0]
                        if val != 'All Zeroes':
                            new_rows[val].append(1)
                        for col in protected_classes[i]['dependent_columns']:
                            if col != val:
                                new_rows[col].append(0)
                    else:
                        # Pick from selection mimicking distribution of entire training data
                        while val in protected_classes[i]['privileged_classes']:
                            val = random.choices(train_dists[protected_classes[i]['name']][0], train_dists[protected_classes[i]['name']][1])[0]
                        if val != 'All Zeroes':
                            new_rows[val].append(1)
                        for col in protected_classes[i]['dependent_columns']:
                            if col != val:
                                new_rows[col].append(0)
                else:
                    raise ValueError(f"Protected class type: {protected_classes[i]['type']} not in ['binary', 'continuous', 'categorical']")
        return new_rows

    def _predict_counterfactuals(self, X: pd.DataFrame):
        if self.verbose:
            print("Predicting Counterfactuals")
        counterfactual_preds = pd.DataFrame()
        for expert in tqdm(self.experts, disable=not self.verbose):
            counterfactual_preds[expert['name']] = expert['model'].predict(X)
        counterfactual_preds.index = X.index
        return counterfactual_preds

    def _cf_fair_models(self, expert_preds: pd.DataFrame, counterfactual_preds: pd.DataFrame):
        '''
        Determines which experts have a maximum consistency score (i.e. are counterfactually fair)

        expert_preds (pd.DataFrame): predictions of all experts for all test data X
        counterfactual_preds (pd.DataFrame): predictions of all counterfactuals generated for all instances in test data X
        '''
        fair_models = {}
        fair_models['cf_fair'] = {}
        if self.verbose:
            print("Getting Counterfactually Fair Models")
        for id in tqdm(expert_preds.index, disable=not self.verbose):
            max = 0
            best_models = []
            for expert in expert_preds:
                cf_preds = counterfactual_preds.loc[id, expert]
                exp_pred = expert_preds.loc[id, expert]
                denom = len(cf_preds) if isinstance(cf_preds, pd.Series) else 1
                cohesion_score = (cf_preds == exp_pred).sum()/denom
                if cohesion_score > max:
                    max = cohesion_score
                    best_models = [expert]
                elif cohesion_score == max:
                    best_models.append(expert)
            fair_models['cf_fair'][id] = best_models
        return pd.DataFrame.from_dict(fair_models)

    def _largest_gap_group_fair_models(self, X: pd.DataFrame, expert_preds: pd.DataFrame, protected_classes: dict) -> pd.Series:
        '''
        DEPRECATED
        '''
        experts = expert_preds.columns
        X = X.copy()
        fair_models = {}
        if self.verbose:
            print("Getting Group Fair Models")
        for pc in tqdm(protected_classes, disable=not self.verbose):
            sps = {}
            if pc['type'] == 'binary':
                pc_col = f'privileged_{pc["name"]}'
                X[pc_col] = X[pc["name"]]
            elif pc['type'] == 'continuous':
                pc_col = f'privileged_{pc["name"]}'
                X[pc_col] = X[pc['name']].apply(lambda x: int((x >= pc['privileged_range_min']) & (x <= pc['privileged_range_max']))) 
            elif pc['type'] == 'categorical':
                pc_col = f'privileged_{pc["name"]}'
                X[pc_col] = X[pc['privileged_classes']].sum(axis=1)
            
            for expert in experts:
                sp = statistical_parity(expert_preds.loc[:, expert], X, pc_col)
                sps[expert] = abs(sp)
            # Find the biggest gap between models and use that as "fair" cutoff
            # this can be cleaned up a lot if too slow but should work for first pass
            sps = pd.DataFrame.from_dict(sps, orient='index')
            sps = sps.sort_values(0)
            cutoff_idx = 0
            max_diff = 0
            for i in range(len(sps) - 1):
                diff = abs(sps.iloc[i+1]) - abs(sps.iloc[i])
                if diff[0] >= max_diff:
                    max_diff = diff[0]
                    cutoff_idx = i
            fair_models[pc['name']] = set(sps.iloc[ :cutoff_idx + 1].index)
        X['group_fair'] = [list(experts)] * X.shape[0]
        for pc in protected_classes:
            X['group_fair'] = X.apply(lambda x: list((set(x['group_fair']) & fair_models[pc['name']])) 
                                      if x[f'privileged_{pc["name"]}'] == 1 
                                      else x['group_fair'], axis=1)
        
        return pd.Series(X['group_fair'])
    
    def _pos_neg_group_fair_models(self, X: pd.DataFrame, expert_preds: pd.DataFrame, protected_classes: dict) -> pd.Series:
        '''
        DEPRECATED
        '''
        experts = expert_preds.columns
        X = X.copy()
        fair_models = {}
        if self.verbose:
            print("Getting Group Fair Models")
        for pc in tqdm(protected_classes, disable=not self.verbose):
            fair_models[pc['name']] = {}
            fair_models[pc['name']][1] = [] 
            fair_models[pc['name']][0] = [] 
            if pc['type'] == 'binary':
                pc_col = f'privileged_{pc["name"]}'
                X[pc_col] = X[pc["name"]]
            elif pc['type'] == 'continuous':
                pc_col = f'privileged_{pc["name"]}'
                X[pc_col] = X[pc['name']].apply(lambda x: int((x >= pc['privileged_range_min']) & (x <= pc['privileged_range_max']))) 
            elif pc['type'] == 'categorical':
                pc_col = f'privileged_{pc["name"]}'
                X[pc_col] = X[pc['privileged_classes']].sum(axis=1)
            
            for expert in experts:
                sp = statistical_parity(expert_preds.loc[:, expert], X, pc_col)
                if sp >= 0:
                    fair_models[pc['name']][1].append(expert)
                if sp <= 0:
                    fair_models[pc['name']][0].append(expert)
        X['group_fair'] = [list(experts)] * X.shape[0]
        for pc in protected_classes:
            X['group_fair'] = X.apply(lambda x: list(set(x['group_fair']) & set(fair_models[pc['name']][x[f'privileged_{pc["name"]}']])), axis=1)
        
        return pd.Series(X['group_fair'])
            
    def _assignment_module(self, expert_preds: pd.DataFrame, meta_preds: pd.DataFrame, cf_fair_models: pd.DataFrame, counterfactual_preds: pd.DataFrame):
        '''
        Select the expert to be used for prediction for each instance in the test set

        expert_preds (pd.DataFrame): predictions of all experts for all test data X
        meta_preds (pd.DataFrame): meta-learner predictions for all experts across all test data X (measure probability of an accurate prediction)
        cf_fair_models (pd.DataFrame): models which are counterfactually fair for each instance in test data X
        counterfactual_preds (pd.DataFrame): predictions of all counterfactuals generated for all instances in test data X
        '''
        if self.verbose:
            print("Assigning Best Models")
        interpretable_models = [i['name'] for i in self.experts if i['interpretable']]
        non_interpretable_models = [i['name'] for i in self.experts if not i['interpretable']] 

        # fair_models = pd.concat([cf_fair_models, group_fair_models], axis=1)
        # fair_models = cf_fair_models.copy()
        # meta_preds['fair_models'] = fair_models.apply(lambda x: list(set(x['cf_fair']) & set(x['group_fair'])), axis=1)
        meta_preds['fair_models'] = cf_fair_models.loc[:, 'cf_fair']

        meta_preds['fair_int'] = meta_preds.apply(lambda x: list(set(x['fair_models']) & set(interpretable_models)), axis=1)
        meta_preds['fair_non'] = meta_preds.apply(lambda x: list(set(x['fair_models']) & set(non_interpretable_models)), axis=1)
        meta_preds['best_int'] = meta_preds.apply(lambda x: x[x['fair_int']].astype(float).idxmax() if len(x['fair_int']) > 0 else np.nan, axis=1)
        meta_preds['best_non'] = meta_preds.apply(lambda x: x[x['fair_non']].astype(float).idxmax() if len(x['fair_non']) > 0 else np.nan, axis=1)
        meta_preds['diff'] = meta_preds.apply(lambda x: (x[x['best_non']] if x['best_non'] is not np.nan else 0)  - (x[x['best_int']] if x['best_int'] is not np.nan else 0), axis=1)

        cutoff = max(np.percentile(meta_preds['diff'], (1-self.non_interp_budget)*100), 0)
        expert_preds['selected_model'] = meta_preds.apply(lambda x: x['best_int'] if x['diff'] <= cutoff else x['best_non'], axis=1)
        # Gotta handle the edge case
        if expert_preds.loc[expert_preds.loc[:, 'selected_model'].isna(), :].shape[0] > 0:
            expert_preds = self._handle_edge_case(expert_preds, meta_preds, counterfactual_preds)
        self.meta_preds = meta_preds
        self.expert_preds = expert_preds
        preds = expert_preds.apply(lambda x: x[x['selected_model']], axis=1)
        return preds
    
    def _handle_edge_case(self, expert_preds: pd.DataFrame, meta_preds: pd.DataFrame, counterfactual_preds: pd.DataFrame):
        '''
        Handles edge case where no interpretable models are counterfactually fair but entire non-interpretable budget has been spent.
        Recalculates counterfactually fair models only considering interpretable experts and then assigns the fair interpretable expert with the maximum meta-learner prediction

        expert_preds (pd.DataFrame): predictions of all experts for all test data X
        meta_preds (pd.DataFrame): meta-learner predictions for all experts across all test data X (measure probability of an accurate prediction)
        counterfactual_preds (pd.DataFrame): predictions of all counterfactuals generated for all instances in test data X
        '''
        if self.verbose:
            print("Handling edge cases with cf fairness")
        
        interpretable_models = [i['name'] for i in self.experts if i['interpretable']]
        cf_fair_models = {}
        cf_fair_models['cf_fair'] = {}

        # Find fairest interpretable models
        for id, _ in tqdm(expert_preds.loc[expert_preds.loc[:, 'selected_model'].isna(), :].iterrows(), disable=not self.verbose):
            max = 0
            best_models = []
            for expert in interpretable_models:
                cf_preds = counterfactual_preds.loc[id, expert]
                exp_pred = expert_preds.loc[id, expert]
                denom = len(cf_preds) if isinstance(cf_preds, pd.Series) else 1
                cohesion_score = (cf_preds == exp_pred).sum()/denom
                if cohesion_score > max:
                    max = cohesion_score
                    best_models = [expert]
                elif cohesion_score == max:
                    best_models.append(expert)
            cf_fair_models['cf_fair'][id] = best_models

        # Do model selection from the fairest interpretable models
        cf_fair_models = pd.Series(cf_fair_models['cf_fair'])
        meta_preds.loc[expert_preds.loc[:, 'selected_model'].isna(), 'fair_int'] = cf_fair_models.loc[expert_preds.loc[:, 'selected_model'].isna()]
        expert_preds.loc[expert_preds.loc[:, 'selected_model'].isna(), 'selected_model'] = meta_preds.loc[expert_preds.loc[:, 'selected_model'].isna(), :].apply(lambda x: x[x['fair_int']].astype(float).idxmax() if len(x['fair_int']) > 0 else np.nan, axis=1)
        
        return expert_preds


if __name__ == '__main__':
    import DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X, y = DataLoader.get_adult_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    pcs = [{'type': 'continuous', 'name': 'age', 'privileged_range_min': 25, 'privileged_range_max': 60}, {'type': 'binary', 'name': 'sex__Male'}, {'type': 'categorical', 'dependent_columns': ['race__Asian-Pac-Islander', 'race__Black', 'race__Other', 'race__White'], 'privileged_classes': ['race__White'], 'name': 'race'}]
    fm = FairMOE(seed=5)
    fm.fit(X_train, y_train)
    preds = fm.predict(X_test, pcs)
    print(accuracy_score(preds, y_test))
