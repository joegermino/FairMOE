import pandas as pd
from scipy.io import arff

def _str_replace(string):
    string = string.replace('<', '_lt_')
    string = string.replace('[', '_')
    string = string.replace(']', '_')
    string = string.replace(' ', '_')
    return string
    
def get_dutch_census_data():
    '''
    ref:
    Van der Laan, P. (2000). The 2001 census in the netherlands. In Conference the Census of Population

    Le Quy, T., Roy, A., Friege, G., & Ntoutsi, E. (2021). Fair-capacitated clustering. 
    In Proceedings of the 14th International Conference on Educational Data Mining (EDM21). (pp. 407-414).
    '''
    data = arff.loadarff('data/dutch_census/dutch_census_2001.arff')
    df = pd.DataFrame(data[0])
    for col in df.columns:
        df[col] = df[col].apply(lambda x: int(x))
        df = pd.get_dummies(df, columns=[col], prefix = [col], drop_first=True)
    X = df.iloc[:, :50]
    X.columns = [_str_replace(col) for col in X.columns]
    y = df['occupation_549']
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_adult_data():
    '''
    ref:
    Becker,Barry and Kohavi,Ronny. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.
    '''
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
            'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 
            'capital_loss', 'hours_per_week', 'native_country', 'y']
    data = pd.read_csv('data/adult/adult.data', na_values=[' ?'], names=cols)
    data2 = pd.read_csv('data/adult/adult.test', na_values=[' ?'], names=cols)
    data = pd.concat((data, data2), axis=0)
    data = data.dropna()
    data['y'] = data['y'] == ' >50K'
    data['y'] = data['y'].apply(lambda x: int(x))
    data = data.reset_index(drop=True)
    data = data.drop('fnlwgt', axis=1)
    data = data.drop('education_num', axis=1) # This step was not advised in the survey paper but this column appears to be repetitve of education so not sure why I'd keep it?
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    for col in cat_cols:
        data = pd.get_dummies(data, columns=[col], prefix = [col], drop_first=True)
    y = data['y']
    X = data.drop('y', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_german_credit_data():
    '''
    ref:
    Hofmann,Hans. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
    '''
    cols = ['status_of_existing_checking_account', 'duration_in_month', 'credit_history', 'purpose',
    'credit_amount', 'savings_account_bonds', 'present_employment_since', 
    'installment_rate_in_percentage_of_disposable_income', 'personal_status_and_sex',
    'other_debtors_guarantors', 'present_residence_since', 'property', 'age_in_years',
    'other_installment_plans', 'housing', 'number_of_existing_credits_at_this_bank',
    'job', 'number_of_people_being_liable_to_provide_maintenance_for', 'telephone', 'foreign_worker', 'y']
    data = pd.read_table('data/german_credit_data/german.data', sep=' ',  names=cols)
    data['y'] = data['y'] == 1
    data['y'] = data['y'].apply(lambda x: int(x))
    data['marital_status'] = data['personal_status_and_sex'].apply(lambda x: int(x in ['A91', 'A92']))
    data['sex'] = data['personal_status_and_sex'].apply(lambda x: int(x in ['A91', 'A93', 'A94']))
    data = data.drop('personal_status_and_sex', axis=1)
    cat_cols = ['status_of_existing_checking_account', 'credit_history', 'purpose',
    'savings_account_bonds', 'present_employment_since', 'other_debtors_guarantors', 'property',
    'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker']
    for col in cat_cols:
        data = pd.get_dummies(data, columns=[col], prefix = [col], drop_first=True)
    y = data['y']
    X = data.drop('y', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_bank_marketing_data(): # This does not touch bank-additional since that is what the survey paper appears to do 
    '''
    ref:
    Moro,S., Rita,P., and Cortez,P.. (2012). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
    '''
    data = pd.read_table('data/bank_marketing/bank-full.csv', sep=';')
    data['y'] = data['y'] == 'yes'
    data['y'] = data['y'].apply(lambda x: int(x))
    
    data['default'] = data['default'] == 'yes'
    data['default'] = data['default'].apply(lambda x: int(x))

    data['housing'] = data['housing'] == 'yes'
    data['housing'] = data['housing'].apply(lambda x: int(x))

    data['loan'] = data['loan'] == 'yes'
    data['loan'] = data['loan'].apply(lambda x: int(x))
    
    cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    for col in cat_cols:
        data = pd.get_dummies(data, columns=[col], prefix = [col], drop_first=True)
    
    y = data['y']
    X = data.drop('y', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_credit_card_data():
    '''
    ref:
    Yeh,I-Cheng. (2016). default of credit card clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.
    '''
    data = pd.read_csv('data/credit_card_clients/default of credit card clients.csv')
    data['SEX'] = data['SEX'] == 1
    data['SEX'] = data['SEX'].apply(lambda x: int(x))
    cat_cols = ['EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for col in cat_cols:
        data = pd.get_dummies(data, columns=[col], prefix = [col], drop_first=True)
    y = data['y']
    X = data.drop('y', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_oulad_data():
    '''
    ref:
    Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. 
    Data 4:170171 doi: 10.1038/sdata.2017.171 (2017).
    '''
    data = pd.read_csv('data/oulad/studentInfo.csv')
    data['gender'] = (data['gender'] == 'M').astype(int)
    data['disability'] = (data['disability'] == 'Y').astype(int)
    data = data.drop('id_student', axis=1)
    data = data.dropna()

    cat_cols = ['code_module', 'code_presentation', 'region', 'highest_education', 'imd_band', 'age_band']
    data = pd.get_dummies(data, columns=cat_cols, prefix=cat_cols, drop_first=True)

    data = data[(data['final_result'] == 'Pass') | (data['final_result'] == 'Fail') | (data['final_result'] == 'Distinction')]
    data['final_result'] = ((data['final_result'] == 'Pass') | (data['final_result'] == 'Distinction')).astype(int)

    y = data['final_result']
    X = data.drop('final_result', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def get_lawschool_data():
    '''
    ref:
    Wightman, L. F. (1998). LSAC national longitudinal bar passage study. LSAC research report series.
    
    Le Quy, T., Roy, A., Friege, G., & Ntoutsi, E. (2021). Fair-capacitated clustering. In Proceedings 
    of the 14th International Conference on Educational Data Mining (EDM21). (pp. 407-414).
    '''
    data = pd.read_csv('data/lawschool/law_dataset.csv')
    data['fulltime'] = (data['fulltime'] == 1).astype(int)
    data['race'] = (data['race'] == 'White').astype(int)
    data['male'] = (data['male'] == 1).astype(int)
    data['pass_bar'] = (data['pass_bar'] == 1).astype(int)

    cat_cols = ['fam_inc', 'tier']
    data = pd.get_dummies(data, columns=cat_cols, prefix=cat_cols, drop_first=True)

    y = data['pass_bar']
    X = data.drop('pass_bar', axis=1)
    X.columns = [_str_replace(col) for col in X.columns]
    return X.reset_index(drop=True), y.reset_index(drop=True)
