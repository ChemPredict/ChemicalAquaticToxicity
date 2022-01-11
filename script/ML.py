import pandas as pd
import numpy as np 
import os 
from itertools import product,combinations
import random
from collections import Counter
import time 
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer,accuracy_score,precision_score,recall_score,roc_auc_score,matthews_corrcoef, confusion_matrix,balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn import metrics
from sklearn.metrics import accuracy_score

seed = 30191375
np.random.seed(seed)
random.seed(seed)

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, average="binary")
def auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)
def new_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
def sp(y_true, y_pred):
    cm = new_confusion_matrix(y_true, y_pred)
    return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
def se(y_true, y_pred):
    cm = new_confusion_matrix(y_true, y_pred)
    return cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
def ba(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def performance_function(y_true,y_pred_proba):
    y_pred = y_pred_proba
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    return [auc(y_true,y_pred_proba),recall(y_true,y_pred),sp(y_true,y_pred),accuracy(y_true,y_pred)]

def build_sklearn_model(X, y, method_name):
    model_map = {"svm":SVC, "knn":KNeighborsClassifier,"nn":MLPClassifier,"rf":RandomForestClassifier}
    tuned_parameters = PARAMETERS[method_name]
    method = model_map[method_name]
    if method == SVC:
        grid = GridSearchCV(method(probability=True,random_state=100), 
                                    param_grid=tuned_parameters,
                                     scoring=SCORING_FNC, cv=kf, n_jobs=-1, refit='AUC' )
    elif method == KNeighborsClassifier:
        grid = GridSearchCV(method(), param_grid=tuned_parameters, 
                                    scoring =SCORING_FNC, cv=kf, n_jobs=-1, refit='AUC')
    else:
        grid = GridSearchCV(method(random_state=100), param_grid=tuned_parameters, 
                                    scoring=SCORING_FNC, cv=kf, n_jobs=-1, refit='AUC')
    grid.fit(X, y)
    return grid.best_estimator_ , grid.best_params_	,grid.cv_results_

PARAMETERS = {
    'svm': {'kernel': ['rbf'], 'gamma': 2**np.arange(-1,-15,-2, dtype=float),
            'C':2**np.arange(-5,15,2, dtype=float),
            'class_weight':['balanced'], 'cache_size':[400] },
    'nn': {"learning_rate":["adaptive"],   #learning_rate:{"constant","invscaling","adaptive"}默认是constant
            "max_iter":[10000],
            "hidden_layer_sizes":[(100,),(400,),(600,),(800,),(1000,),(200,100),(200,100,50)],
            "alpha":10.0 ** -np.arange(1, 7),
            "activation":["relu"],  #"identity","tanh","relu","logistic"
            "solver":["adam"],     #"lbfgs" for small dataset
            'warm_start':[True]},
    'knn': {"n_neighbors":range(2,15,1),"weights":['distance'],'p':[1,2],
            'metric':['minkowski','jaccard']},
    'rf': {"n_estimators":range(10,501,20),
            "criterion" : ["gini",'entropy'], #['entropy']
            "oob_score": ["False"],
            "class_weight":["balanced_subsample"]}
}
SCORING_FNC = {'SE':'recall','SP':make_scorer(sp),'AUC':'roc_auc','ACC':'accuracy'}
kf = StratifiedKFold(n_splits = 5,shuffle=True,random_state = 100)


global PADELFPS,METHODS
PADELFPS = ['MACCS', 'CDKExt', 'CDK', 'EState', 'GraphFP', 'PubChemFP', 'SubFP', 'KRFP', 'AP2D']
METHODS = ['svm', 'nn', 'rf', 'knn']

def cansmi2fp(cansmi,Description,json_path = os.path.join(os.getcwd(),'data','FP.json')):
    fp_json = pd.read_json(json_path,'index')
    fp_map = cansmi.map(fp_json[Description])
    converted = []
    for fp in fp_map:
        converted.append([int(i) for i in fp.strip('[]').split(',')])
    return np.array(converted)

def get_performance(cv_res):
    df = pd.DataFrame(cv_res)
    r = df.sort_values(by='rank_test_AUC').iloc[0]
    return r[['mean_test_AUC','mean_test_SE', 'mean_test_SP', 'mean_test_ACC']]


def ML_Model(FilePath):

    try:
        os.mkdir('model')
        os.mkdir('model/temp')
    except Exception:
        pass

    TrainSetPath = FilePath
    TestSetPath = FilePath.replace('Train','Test')
    train = pd.read_csv(TrainSetPath)
    test = pd.read_csv(TestSetPath)

    FileName = FilePath.split('/')[-1]
    for Description,Algorithm,SplitType in product(PADELFPS,METHODS,['random']):
        
        ModelName = f"{FileName.split('_T')[0]}_{Description}_{Algorithm}_{SplitType}.m"
        print(f'ONGOING----> {ModelName}')

        X_train = np.array(cansmi2fp(train.Canonical_Smiles,Description))
        X_test = np.array(cansmi2fp(test.Canonical_Smiles,Description))
        y_train = train.label
        y_test = test.label

        best_model, best_params, cv_res = build_sklearn_model(X_train, y_train, Algorithm)
        model = best_model
        joblib.dump(model,f'model/{ModelName}')

        result = [FileName.split('_T')[0],Description,Algorithm,SplitType,str(best_params)]+get_performance(cv_res).tolist()
        pd.DataFrame([result],columns = ['Endpoint','Description','Algorithm','SplitType','param','AUC','SE','SP','ACC']).to_csv(f'model/temp/{ModelName}_CrossValid.csv',index = False)

        proba = model.predict_proba(X_test)[:,1]
        r = [FileName.split('_T')[0],Description,Algorithm,SplitType]+performance_function(y_test,proba)
        pd.DataFrame([r],columns =['Endpoint','Description','Algorithm','SplitType','AUC','SE','SP','ACC']).to_csv(f'model/temp/{ModelName}_TestValid.csv',index = False)

def ba(FilePath):
    TrainSetPath = FilePath
    TestSetPath = FilePath.replace('Train','Test')
    test = pd.read_csv(TestSetPath)

    FileName = FilePath.split('/')[-1]
    Endpoint = FileName.split('_T')[0]
    for Description,Algorithm,SplitType in product(PADELFPS,METHODS,['random']):
        ModelPath = f'model/{Endpoint}_{Description}_{Algorithm.lower()}_random.m'
        model = joblib.load(ModelPath)

        X_test = np.array(cansmi2fp(test.Canonical_Smiles,Description))
        y_test = test.label

        pred = model.predict(X_test)
        ba = balanced_accuracy_score(y_test,pred)
        pd.DataFrame([[Endpoint,Description,Algorithm,ba]]).to_csv(f'model/temp/BalancedAccuracy.csv',mode = 'a',index = False)
    

def consistent():

    TestSetPath = f'data/GlobalModel_TestSet.csv'
    test = pd.read_csv(TestSetPath)

    BestModels = [('BS','RF','KRFP'),('RT','RF','KRFP'),('FHM','RF','PubChemFP'),('SHM','SVM','CDKExt')]
    compare = pd.DataFrame()
    for Endpoint,Algorithm,Description in BestModels:
        Algorithm = Algorithm.lower()
        X_test = np.array(cansmi2fp(test.Canonical_Smiles,Description))
        y_true = test.label
        ModelPath = f'model/LocalModel_{Endpoint}_{Description}_{Algorithm.lower()}_random.m'
        model = joblib.load(ModelPath)
        pred = model.predict(X_test)
        compare[Endpoint] = pred
    final = compare.sum(axis=1)
    cr = (Counter(final)[0]+Counter(final)[4])/len(final)
    keep = final.replace({0:True,4:True,1:False,2:False,3:False}).tolist()
    cm = metrics.confusion_matrix(y_true[keep],final[keep].replace({4:1}))
    acc = accuracy(y_true[keep],final[keep].replace({4:1}))
    se = cm[1,1]/(cm[1,1]+cm[1,0])
    sp = cm[0,0]/(cm[0,0]+cm[0,1])
    for x in [acc,se,sp,cr]:
        print(np.round(x,3))

def ProbaThreshold(Threshold):
    from collections import Counter
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    TestSetPath = f'data/GlobalModel_TestSet.csv'
    test = pd.read_csv(TestSetPath)

    BestModels = [('RF','PubChemFP'),('RF','KRFP'),('NN','PubChemFP')]
    
    report = []
    for Algorithm,Description in BestModels:
        Algorithm = Algorithm.lower()
        X_test = np.array(cansmi2fp(test.Canonical_Smiles,Description))
        y_true = test.label
        ModelPath = f'model/GlobalModel_{Description}_{Algorithm.lower()}_random.m'
        model = joblib.load(ModelPath)
        proba = model.predict_proba(X_test)
        pred = model.predict(X_test)
        PredMask = [True if (x>=Threshold) or (x<=1-Threshold) else False for x in proba[:,1]]
        pred[pred>=Threshold] =1
        pred[pred<=1-Threshold] = 0
        print(pred,Counter(PredMask))
        cr = Counter(PredMask)[True]/len(PredMask)
        cm = metrics.confusion_matrix(y_true[PredMask],pred[PredMask])
        acc = accuracy(y_true[PredMask],pred[PredMask])
        se = cm[1,1]/(cm[1,1]+cm[1,0])
        sp = cm[0,0]/(cm[0,0]+cm[0,1])

        report.append([Algorithm,Description,Threshold,acc,se,sp,cr])
    print(pd.DataFrame(report))
    return pd.DataFrame(report)

def EVData(Description):
    df = pd.read_csv('data/external_validation.csv')
    y = df['class-species']
    fp_data = pd.read_json('data/external_validation_FP.json','index')
    fp_map = df.e_cid.map(fp_data[Description])

    X = np.array(fp_map.tolist())
    return X,y

def ExternalValid():
    
    BestModels = [('RF','PubChemFP',0.7),('RF','KRFP',0.7),('NN','PubChemFP',0.7)]
    
    report = []
    for Algorithm,Description,Threshold in BestModels:
        Algorithm = Algorithm.lower()
        X_test,y_true = EVData(Description)
        ModelPath = f'model/GlobalModel_{Description}_{Algorithm.lower()}_random.m'
        model = joblib.load(ModelPath)
        proba = model.predict_proba(X_test)
        pred = model.predict(X_test)
        PredMask = [True if (x>=Threshold) or (x<=1-Threshold) else False for x in proba[:,1]]
        pred[pred>=Threshold] =1
        pred[pred<=1-Threshold] = 0
        print(pred,Counter(PredMask))
        cr = Counter(PredMask)[True]/len(PredMask)
        cm = metrics.confusion_matrix(y_true[PredMask],pred[PredMask])
        AUC = auc(y_true[PredMask],proba[PredMask][:,1])
        acc = accuracy(y_true[PredMask],pred[PredMask])
        se = cm[1,1]/(cm[1,1]+cm[1,0])
        sp = cm[0,0]/(cm[0,0]+cm[0,1])

        report.append([Algorithm,Description,Threshold,AUC,acc,se,sp,cr])
    print(pd.DataFrame(report))
    return pd.DataFrame(report)



DataDir = os.path.join(os.getcwd(),'data')
        
'''
for FileName in os.listdir(DataDir):
    if ('Model' in FileName) and ('TrainSet' in FileName):
        ML_Model(os.path.join(DataDir,FileName))
'''
if __name__ == '__main__':
    from multiprocessing import Pool
    
    for FileName in os.listdir(DataDir):
        if ('Model' in FileName) and ('TrainSet' in FileName):
            ba(os.path.join(DataDir,FileName))
'''
CV = pd.DataFrame()
TV = pd.DataFrame()
for FileName in os.listdir(f'model/temp'):
    if 'CrossValid' in FileName:
        CV = pd.concat([CV,pd.read_csv(os.path.join(os.getcwd(),'model','temp',FileName))])
    elif 'TestValid' in FileName:
        TV = pd.concat([TV,pd.read_csv(os.path.join(os.getcwd(),'model','temp',FileName))])
    else:# for future, external validation
        pass

CV.to_csv(os.path.join(os.getcwd(),'data','ML_ST_CrossValid.csv'),index= False)
TV.to_csv(os.path.join(os.getcwd(),'data','ML_ST_TestValid.csv'),index= False)
'''
'''
# node-parallel
for FileName in os.listdir(DataDir):
    if ('Model' in FileName) and ('TrainSet' in FileName) and ('Global' in FileName):
        ML_Model(os.path.join(DataDir,FileName))
'''
'''
Table5 = pd.DataFrame()
for t in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
    Table5 = pd.concat([Table5,ProbaThreshold(t)])
Table5.to_csv('Table5.csv')

'''
'''
ExternalValid().to_csv('Table7.csv')
'''