import pandas as pd
# import numpy as np
# import datetime
# from collections import Counter
#
# #ML Libraries
# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn import preprocessing
# from sklearn.model_selection import cross_validate
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
#
# #Feature extraction
# from tsfresh import extract_relevant_features
#
# X = pd.read_csv('ADM-BAN-2001-0_Sem1_2_3_4_5_6_7_8_Var_grau_features.csv', sep=';' )
# # print(len(X))
# # X.to_csv('X.csv',header=True)
# y_read = pd.read_csv('ADM-BAN-2001-0_Sem1_2_3_4_5_6_7_8_Var_grau_labels.csv', sep=';', header = None , names = ['id','class'], index_col = 'id' ).to_dict('index')
# print(y_read)
# y = []
# for alumno in X['id']:
#     y.append(y_read[alumno]['class'])
# print(y)
#
# X_selected = X.drop(['id'], axis=1)
#
# print ("Dimensiones antes del sampling: ")
# print (Counter(y).items()) #Before Sampling
# #X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_sample(X_selected, y)
# X_resampled, y_resampled = SMOTE(random_state=42).fit_sample(X_selected, y)
# print ("Dimensiones despues del sampling: ")
# print (Counter(y_resampled).items()) #After Sampling
# data=pd.concat([X_resampled.assign(drop=i) for i in y_resampled],ignore_index=False)

a=pd.read_csv('historicosFinal.csv',sep=';')
print(len(a.columns))
print(len(a['sit_vinculo_atual'].unique()))