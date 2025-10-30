from warnings import simplefilter
simplefilter(action="ignore",category=UserWarning)
import argparse
parser = argparse.ArgumentParser(description='Binary Classification')
parser.add_argument('--filein',help='input matrix',required=True) 
parser.add_argument('--feature',help='feature file',required=True) 
parser.add_argument('--outpath',required=True) 
argv = vars(parser.parse_args())

# coding:utf8
import random, re, os, pprint
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

random.seed(100)

isExists=os.path.exists(argv['outpath'])
if not isExists:
    os.mkdir(argv['outpath'])

arryposi=['SampleID','Type']

with open(argv['feature'],'r') as fileposi:
    next(fileposi)
    for line in fileposi:
        tmp = line.strip().split('\t')
        arryposi.append(tmp[0])

df = pd.read_csv(argv['filein'],header=0,sep='\t')
'''
filein format:
SampleID    Type    Cohort    feature1    feature2    ...    featuren
Sample1    COREAD    TrainSet    0.9281437125748504    0.9233716475095786    ...    0.9106840022611644
'''
if not set(df['Type']) <= set(['T','N']) :   
    df['Type'] = df['Type'].replace('Healthy','N').replace('COREAD','T').replace('ESCA','T').replace('LIHC','T').replace('Lung','T').replace('STAD','T').replace('THCA','T').replace('OV','T')

train = df[(df['Cohort']=='TrainSet')][arryposi]
features = train.columns[2:]
test = df[(df['Cohort']=='TestSet')][arryposi]

y = train['Type']
arrytitle = np.array(features)

models = {
    'LogisticRegression':
    LogisticRegression(random_state=1,
                       class_weight="balanced",
                       multi_class='ovr',
                       max_iter=5000),
}

params = {
    'LogisticRegression': {
        'clf__solver': ['newton-cg', 'sag', 'lbfgs'],
        'clf__C': [1e-3, 1e-4, 0.5, 1, 2, 3]
    }
}

name = 'LogisticRegression'
est = models[name]
est_params = params[name]
pipe=Pipeline([('scl',StandardScaler()),('clf',est)])
gs=GridSearchCV(estimator=pipe,param_grid=est_params,scoring='f1',cv=5,n_jobs=-1,error_score='raise')
gs.fit(train[features].astype(np.float64),train['Type'].replace('N',0).replace('T',1))
clf=gs.best_estimator_

# #===============================================
# #    LogisticRegression Features' weights
# #===============================================
filecoef = open(argv['outpath']+'/result.coef.LogisticRegression.xls', 'w') 
tmp = {}
for i in range(len(list(arrytitle))):
    if clf['clf'].coef_[0][i] != 0:
        tmp[arrytitle[i]] = clf['clf'].coef_[0][i]
for k,v in sorted(tmp.items(),key=lambda x:x[1],reverse=True):
    filecoef.write(str(k)+'\t'+str(v) + '\n')
filecoef.write('intercept'+'\t'+str(clf['clf'].intercept_[0]))

# #========================
# #       Training
# #========================
probas_ = clf.predict_proba(train[features].astype(np.float64))
train_pred = []
fpr, tpr, thresholds = roc_curve(train['Type'], probas_[:, 1], pos_label='T')
roc_auc = auc(fpr, tpr)
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc['sum'] = roc['tpr'] + roc['1-fpr']
roc_t = roc.loc[roc['1-fpr']>=0.95]
roc_t = roc_t.loc[roc_t['sum']==roc_t['sum'].max()]
roc_t = roc_t.loc[roc_t['1-fpr']==roc_t['1-fpr'].max()]
best_thresholds = roc_t['thresholds'].tolist()[0]

train_pred = []
train_proba = []
for row in probas_[:, 1].tolist():
    train_proba.append(row)
    if row >= best_thresholds:
        train_pred.append("T")
    else:
        train_pred.append("N")
mat = np.transpose(confusion_matrix(train['Type'], train_pred, labels=["T", "N"]))
print (name + ' TrainSet ...')
print (mat)
print (classification_report(train['Type'], train_pred,digits=3))
print ('AUC: '+str('%.3f'%roc_auc))
fig=plt.figure(figsize=(6,8))
ax = fig.add_subplot(211)
lw = 2
ax.plot(fpr, tpr, color='darkorange',lw=lw, label='AUC:%0.3f,Thresholds:%0.3f,\nTPR:%0.3f,FPR:%0.3f' % (roc_auc,roc_t['thresholds'],roc_t['tpr'],roc_t['fpr']))
ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(name+' TrainSet')
ax.legend(loc="lower right")
fileout_train = open(argv['outpath']+f'/result.train.{name}.xls', 'w') 
fileout_train.write('SampleID'+'\t'+'Type'+'\t'+'Predict'+'\t'+'Nprob'+'\t'+'Tprob'+'\n')
for i in range(0,len(train_pred)):
    fileout_train.write(re.search('\s+(\S+)\nName',str(train['SampleID'][i:i+1])).group(1)+'\t'+re.search('\s+(\S+)\nName',str(train['Type'][i:i+1])).group(1)+'\t'+train_pred[i]+'\t'+'\t'.join(str(j) for j in probas_.tolist()[i])+'\n')
fileout_train.close()

#========================
#       Test
#========================
probas_ = clf.predict_proba(test[features].astype(np.float64))
protype_ = clf.predict(test[features].astype(np.float64))
test_pred = []
fpr, tpr, thresholds = roc_curve(test['Type'], probas_[:, 1], pos_label='T')
roc_auc = auc(fpr, tpr)
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc['sum'] = roc['tpr'] + roc['1-fpr']
roc_t = roc.loc[roc['1-fpr']>=0.95]
roc_t = roc_t.loc[roc_t['sum']==roc_t['sum'].max()]
roc_t = roc_t.loc[roc_t['1-fpr']==roc_t['1-fpr'].max()]

test_pred = []
test_proba = []
for row in probas_[:, 1].tolist():
    test_proba.append(row)
    if row >= best_thresholds:
            test_pred.append("T")
    else:
            test_pred.append("N")
mat = np.transpose(confusion_matrix(test['Type'], test_pred, labels=["T", "N"]))
ax = fig.add_subplot(212)
lw = 2
ax.plot(fpr, tpr, color='darkorange',lw=lw, label='AUC:%0.3f,Thresholds:%0.3f,\nTPR:%0.3f,FPR:%0.3f' % (roc_auc,roc_t['thresholds'],roc_t['tpr'],roc_t['fpr']))
ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(name+' TestSet')
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(argv['outpath']+'/result.'+name+'.pdf',dpi=300)
print (name + ' Validation ...')
print (mat)
print (classification_report(test['Type'], test_pred,digits=3))
print ('AUC: '+str('%.3f'%roc_auc))
fileout = open(argv['outpath']+f'/result.test.{name}.xls', 'w') 
fileout.write('SampleID'+'\t'+'Type'+'\t'+'Predict'+'\t'+'Nprob'+'\t'+'Tprob'+'\n')
for i in range(0,len(protype_)):
    fileout.write(re.search('\s+(\S+)\nName',str(test['SampleID'][i:i+1])).group(1)+'\t'+re.search('\s+(\S+)\nName',str(test['Type'][i:i+1])).group(1)+'\t'+test_pred[i]+'\t'+'\t'.join(str(j) for j in probas_.tolist()[i])+'\n')
fileout.close()