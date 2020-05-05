#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:11:36 2019

@author: paulina
"""

import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import os


os.chdir('D:/GEOL_SUSC')
proba_dict_reg = defaultdict(list)

wells_df = pd.read_csv('WELLS_INPUT.csv')

wells_df['montney_part'] = 0.0

wells_df['montney_part'].loc[(wells_df['Depth_factor'] > 0) &
                             (wells_df['Depth_factor'] < 0.34)] = 'upper'
wells_df['montney_part'].loc[(wells_df['Depth_factor'] >= 0.34) &
                             (wells_df['Depth_factor'] < 0.67)] = 'middle'
wells_df['montney_part'].loc[(wells_df['Depth_factor'] >= 0.67) &
                             (wells_df['Depth_factor'] < 1)] = 'lower'

wells_df = wells_df[(wells_df['Depth_factor'] > 0) &
                    (wells_df['Depth_factor'] <1)]

wells_df = wells_df[(wells_df['well_TVD'] > wells_df['Montney_top_depth'])]

wells_df = wells_df[(wells_df['Montney_top_depth'] < wells_df['Debolt_top_depth']) &
                    (wells_df['Debolt_top_depth'] < wells_df['Precambrian_top_depth'])]

wells_count = wells_df.groupby(['montney_part', 'seismogenic']).agg({'UWI': 'count'})
seis_wells = wells_df[wells_df['seismogenic'] == 1]
non_seis_wells = wells_df[wells_df['seismogenic'] == 0]

wells_ratios_num = wells_count.groupby(level=[0, 1]).sum()
wells_ratios = wells_count.groupby(level=0).apply(lambda x:
                                                  100 * x / float(x.sum()))

print(wells_ratios_num)
print(np.around(wells_ratios, decimals=2))

print("all wells", wells_df.shape[0])
print("seismogenic wells:", wells_df[wells_df.seismogenic == 1].shape[0])
print("non-seismogenic wells", wells_df[wells_df.seismogenic == 0].shape[0])

seismo_wells_percent = np.around(
    wells_df[wells_df.seismogenic == 1].shape[0] / wells_df[wells_df.seismogenic == 0].shape[0], decimals=2)

print('Percentage of seismogenic wells {0:.2%}'.format(seismo_wells_percent))

wells_df = wells_df[['UWI','Proximity_to_Cordilleran_belt','Pressure_gradient','Montney_thickness',
                     'Montney_top_depth','Shmax_azimuth_variance', 'Depth_factor','Proximity_to_faults',
                     'Vert_distance_to_Precambrian','Vert_distance_to_Debolt','seismogenic']]
wells_df.reset_index(inplace=True, drop=True)


X = wells_df[['Proximity_to_Cordilleran_belt',
              'Proximity_to_faults',
              'Pressure_gradient',
              'Vert_distance_to_Precambrian',
              'Depth_factor',
              'Vert_distance_to_Debolt',
              'Shmax_azimuth_variance']]

size= (X.shape[1],)  # create the vector to store feature importance values
feature_importance_sum = np.zeros(size)
feat_importance_dict = defaultdict()

y = wells_df['seismogenic']

scaler = StandardScaler()
reg = LogisticRegression(class_weight='balanced',random_state=42)

conf_matrix_sum = np.zeros([2,2])

n_splits = 10

split = ShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=42)
i=0
tprs = []
aucs = []
precs=[]
precs0 = []
recs=[]
recs0=[]
mean_fpr = np.linspace(0, 1, 100)
k=0

fig, (ax1,ax2) = plt.subplots(2,1, figsize=(8,4))

plt.subplots_adjust(bottom=-0.8, right=0.8, top=1.5)
feat_import_df = pd.DataFrame(columns=X.columns)

for train_index, test_index in split.split(X, y):

    X_train = X.loc[train_index].values
    X_test = X.loc[test_index].values
    y_train = y.loc[train_index].values
    y_test = y.loc[test_index].values

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    reg.fit(X_train_scaled, y_train)

    y_pred_proba = reg.predict_proba(X_test_scaled)
    y_pred = reg.predict(X_test_scaled)

    for z in test_index:
        key = wells_df.UWI[z]
        proba_dict_reg[key].append(y_pred_proba[list(test_index).index(z)][1])

    feature_importance = abs(reg.coef_[0])

    for index, feature in enumerate(feature_importance):
        feat_import_df.loc[k, X.columns[index]] = feature

    feature_importance_sum = feature_importance_sum + feature_importance

    #     Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    prec, rec, fbeta_score, supp = precision_recall_fscore_support(y_test,y_pred)

    precs.append(prec[1])
    precs0.append(prec[0])
    recs.append(rec[1])
    recs0.append(rec[0])

    conf_matrix_sum = conf_matrix_sum + confusion_matrix(y_test, y_pred)

    k = k + 1

to_plot = zip(feat_import_df.columns, feat_import_df.mean(), feat_import_df.std())

to_plot = list(to_plot)
res = sorted(to_plot, key = lambda x: x[1], reverse=False)
res_df = pd.DataFrame(res, columns=('feat', 'mean', 'std'))

y_pos = np.arange(res_df.shape[0])
ax1.barh(y_pos, res_df['mean'], xerr=res_df['std'], align='center', alpha=0.4, ecolor='#686D75', capsize=5)
ax1.set_xlabel('Mean feature importance')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(res_df.feat)
ax1.xaxis.grid(True)

extent_feat_importance = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('FEAT_IMPORTANCE.png', bbox_inches=extent_feat_importance.expanded(1.9, 1.3), dpi=300)
# plt.show()

dict_logreg = pd.DataFrame(proba_dict_reg.items(), columns=['UWI', 'proba_seismogenic_logreg'])
lst_col = 'proba_seismogenic_logreg'
dict2_logreg = pd.DataFrame({
    col: np.repeat(dict_logreg[col].values, dict_logreg[lst_col].str.len())
    for col in dict_logreg.columns.drop(lst_col)}).assign(**{lst_col: np.concatenate(dict_logreg[lst_col].values)})[
    dict_logreg.columns]

mean_probabs_logreg = dict2_logreg.groupby('UWI').mean()
wells_probas = wells_df.merge(mean_probabs_logreg, on='UWI')


mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax2.plot(mean_fpr, mean_tpr, color='b',
         lw=1.5, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax2.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4,
                 label=r'$\pm$ 1 std. dev. (AUC = %0.2f)' % mean_auc)

ax2.plot([0, 1], [0, 1], color='red', lw=1.5, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Receiver operating characteristic curve')
ax2.legend(loc="lower right")
plt.show()

extent_ROC = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('ROC_curve.png', bbox_inches=extent_ROC.expanded(1.3, 1.2), dpi=300)

TP = conf_matrix_sum[1,1]
FP = conf_matrix_sum[1,0]
FN = conf_matrix_sum[0,1]
TN = conf_matrix_sum[0,0]

hand_precision = TP/ (TP + FP)
hand_recall = TP / (TP + FN)
hand_tpr = TP / (TP + FN)
hand_fpr = FP / (FP + TN)

print("mean precision ", sum(precs) / len(precs))
print("mean recall", sum(recs) / len(recs))

mean_conf_matrix = conf_matrix_sum / n_splits
print(conf_matrix_sum)
print("average confusion matrix:", mean_conf_matrix.astype(int))
mean_feature_importance = feature_importance_sum/n_splits
sorted_idx = np.argsort(mean_feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

d_logreg = pd.DataFrame(proba_dict_reg.items(), columns=['UWI', 'proba_seismogenic_logreg'])
lst_col = 'proba_seismogenic_logreg'
d2_logreg = pd.DataFrame({
      col:np.repeat(d_logreg[col].values, d_logreg[lst_col].str.len())
      for col in d_logreg.columns.drop(lst_col)}
    ).assign(**{lst_col:np.concatenate(d_logreg[lst_col].values)})[d_logreg.columns]

mean_probabs_logreg = d2_logreg.groupby('UWI').mean()
wells_probas = wells_df.merge(mean_probabs_logreg, on='UWI')
print("WELLS PROBAS SHAPE", wells_probas.shape)

# plt.show()

# wells_probas.to_csv('wells_class_probabilities.csv')

