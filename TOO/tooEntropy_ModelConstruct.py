from warnings import simplefilter
simplefilter(action="ignore",category=UserWarning)

import argparse
parser = argparse.ArgumentParser(description='TOO Classification')
parser.add_argument('--filein',help='input matrix',required=True) 
parser.add_argument('--feature',help='feature file',required=True) 
parser.add_argument('--outpath',required=True) 
argv = vars(parser.parse_args())

import random, re, os, pprint, csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import joblib
from keras import utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
pd.set_option('display.width', 100000)
import seaborn as sns

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
SampleID    Type    Cohort    feature1    feature2    ...    featuren    Sex
Sample1    COREAD    TrainSet    0.9281437125748504    0.9233716475095786    ...    0.9106840022611644    Female
'''
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'Female':0,'Male':1})
train = df[(df['Cohort']=='TrainSet')][arryposi]
test = df[(df['Cohort']=='TestSet')][arryposi]

features = train.columns[2:]
arrytitle = np.array(features)
y = train['Type']
typeclass = list(set(y))
typeclass.sort()


models = {
    'LogisticRegression':
    LogisticRegression(random_state=1,
                       multi_class='ovr',
                       max_iter=5000,
                       penalty='elasticnet')
}

params = {
    'LogisticRegression': {
        'clf__estimator__solver': ['saga'],
        'clf__estimator__C': [1e-3, 1e-4, 1e-5, 0.5],
        "clf__estimator__l1_ratio":np.arange(0, 1, 0.1)
    }
}

name = 'LogisticRegression'
est = models[name]
est_params = params[name]
pipe_svc = Pipeline([('scl', StandardScaler()),
                    ('clf', OneVsRestClassifier(est))])
gs = GridSearchCV(estimator=pipe_svc,
                param_grid=est_params,
                scoring="balanced_accuracy",
                cv=5,
                n_jobs=-1,
                error_score='raise',
                verbose=False)
gs.fit(train[features].astype(np.float64), y)
clf = gs.best_estimator_

#========================
#       Training
#========================

probas_ = clf.predict_proba(train[features].astype(np.float64))
protype_ = clf.predict(train[features].astype(np.float64))
train_pred = []
for i in protype_:
    train_pred.append(i)
train['class_pred'] = train_pred
train_proba = []
for row in probas_[:, ].tolist():
    train_proba.append('\t'.join([str(x) for x in row]))
train['%s' % ('\t'.join(typeclass))] = train_proba

#========================
#       top2
#========================
arry_train=[]
temp_train = probas_[:, ].tolist()
for j in range(len(temp_train)):
    class1 = ['COREAD','ESCA','LIHC','Lung','OV','STAD','THCA']
    top2ix = np.argsort(temp_train[j])[-2]
    top2class = class1[top2ix]
    arry_train.append(top2class)
train['top2'] = arry_train

res = train[['SampleID', 'Type', 'class_pred','top2','\t'.join(typeclass)]]
res.to_csv(argv['outpath']+'Top2result.train.'+name+'.xls', index=False, sep='\t',quoting = csv.QUOTE_NONE, escapechar = ' ')

mat = np.transpose(confusion_matrix(train['Type'], train_pred))
print (name + ' TrainSet ...')
print (mat)
print (classification_report(train['Type'], train_pred,digits=3))
top2_acc = len(res.loc[(res['Type']==res['class_pred']) | (res['Type']==res['top2'])])/len(res)
print ('Top2 Accuracy: %.2f%%' % (top2_acc*100))

#Multiclass Confusion Matrix Plot
fig=plt.figure(figsize=(15,10))
ax1=fig.add_subplot(221)
rf_cnf_mat = confusion_matrix(train['Type'], train_pred)
cm_array_df = pd.DataFrame(rf_cnf_mat, index=typeclass, columns=typeclass).T
ax = sns.heatmap(cm_array_df, annot=True, linewidths=4, fmt='g' , cmap='Oranges',ax=ax1,cbar=True)
ax.set_title(name + ' TrainSet', fontsize = 15)
ax.set_ylabel('Predicted Cancer Signal Origin', fontsize = 10)
ax.set_xlabel('Actual Cancer Signal Origin', fontsize = 10)

#Multiclass AUC Plot
ax = fig.add_subplot(222)
encoder = preprocessing.LabelEncoder()
encoded_Y_train = encoder.fit_transform(train['Type'])
y_label_train = utils.to_categorical(encoded_Y_train)
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
for i in range(len(typeclass)):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_label_train[:, i], probas_[:, i])
    n = np.arange(len(tpr[i])) 
    roc_auc[i] = auc(fpr[i], tpr[i])
for i in range(len(typeclass)):
    plt.plot(fpr[i], tpr[i], lw=2, label='{0} - AUC: {1:0.2f}'.format(typeclass[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(name+' TrainSet')
plt.legend(loc="lower right")


#========================
#       Test
#========================
probas_ = clf.predict_proba(test[features].astype(np.float64))
protype_ = clf.predict(test[features].astype(np.float64))
test_pred = []
for i in protype_:
    test_pred.append(i)
test['class_pred'] = test_pred
test_proba = []
for row in probas_[:, ].tolist():
    test_proba.append('\t'.join([str(x) for x in row]))
test['\t'.join(typeclass)] = test_proba


#========================
#       top2
#========================   
arry=[]
temp = probas_[:, ].tolist()
for j in range(len(temp)):
    class1 = ['COREAD','ESCA','LIHC','Lung','OV','STAD','THCA']
    top2ix = np.argsort(temp[j])[-2]
    top2class = class1[top2ix]
    arry.append(top2class)
test['top2'] = arry

res = test[['SampleID', 'Type', 'class_pred','top2','\t'.join(typeclass)]]
res.to_csv(argv['outpath']+'Top2result.test.'+name+'.xls', index=False, sep="\t",quoting = csv.QUOTE_NONE, escapechar = ' ' )

mat = np.transpose(confusion_matrix(test['Type'], test_pred))
print (name + ' TestSet ...')
print (mat)
print (classification_report(test['Type'], test_pred,digits=3))
top2_acc = len(res.loc[(res['Type']==res['class_pred']) | (res['Type']==res['top2'])])/len(res)
print ('Top2 Accuracy: %.2f%%' % (top2_acc*100))

#Multiclass Confusion Matrix Plot
ax1=fig.add_subplot(223)
rf_cnf_mat = confusion_matrix(test['Type'], test_pred)
cm_array_df = pd.DataFrame(rf_cnf_mat, index=typeclass, columns=typeclass).T
ax = sns.heatmap(cm_array_df, annot=True, linewidths=4, fmt='g' , cmap='Oranges',ax=ax1,cbar=True)
ax.set_title(name + ' TestSet', fontsize = 15)
ax.set_ylabel('Predicted Cancer Signal Origin', fontsize = 10)
ax.set_xlabel('Actual Cancer Signal Origin', fontsize = 10)

#Multiclass AUC Plot
ax = fig.add_subplot(224)
encoder = preprocessing.LabelEncoder()
encoded_Y = encoder.fit_transform(test['Type'])
y_label = utils.to_categorical(encoded_Y)
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
for i in range(len(typeclass)):
    fpr[i], tpr[i], _ = roc_curve(y_label[:, i], probas_[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
for i in range(len(typeclass)):
    plt.plot(fpr[i], tpr[i], lw=2, label='{0} - AUC: {1:0.2f}'.format(typeclass[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(name+' TestSet')
plt.legend(loc="lower right")
plt.savefig(argv['outpath']+'AUC.'+name+'.pdf',dpi=300)