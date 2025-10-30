import argparse
parser = argparse.ArgumentParser(description='Feature Selection')
parser.add_argument('--filein', required=True) 
parser.add_argument('--fileout', required=True)
argv = vars(parser.parse_args())

# coding:utf8
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random

random.seed(100)

df = pd.read_csv(argv['filein'],header=0,sep='\t')
'''
filein format:
SampleID    Type    Cohort    feature1    feature2    ...    featuren
Sample1    COREAD    TrainSet    0.9281437125748504    0.9233716475095786    ...    0.9106840022611644
'''
if not set(df['Type']) <= set(['T','N']):
    df['Type'] = df['Type'].replace('Healthy','N').replace('COREAD','T').replace('ESCA','T').replace('LIHC','T').replace('Lung','T').replace('STAD','T').replace('THCA','T').replace('OV','T')
train=df[df['Cohort']=='TrainSet']
features = df.columns[3:]

fileout=open(argv['fileout'],'w')
fileout.write('marker_index'+'\t'+'Importance' + '\n')

scaler = preprocessing.StandardScaler().fit(train[features])
X_train_scale = scaler.transform(train[features])
arryhit = []
arrytitle = np.array(features)

clf = RandomForestClassifier(random_state = 42, class_weight="balanced")
y=train['Type']
clf.fit(X_train_scale, y)
arryflag = list(arrytitle[clf.feature_importances_ != 0])
arryhit = arryhit + arryflag
tmp = {}
for i in range(len(list(arrytitle))):
    if clf.feature_importances_[i] != 0:
        tmp[arrytitle[i]] = clf.feature_importances_[i]
        
for k,v in sorted(tmp.items(),key=lambda x:x[1],reverse=True):
    fileout.write(str(k)+'\t'+str(v) + '\n')

fileout.close()
