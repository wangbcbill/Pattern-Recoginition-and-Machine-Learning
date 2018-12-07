from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

#2.3
#senator model
vote_diff_sen_true=np.zeros(int(n/2))
for i in range(n):
    if i%2==0 and i%4==0:
        vote_diff_sen_true[int(i/2)]=vote_diff_sen[i]
    if i%2==0 and i%4==2:
        vote_diff_sen_true[int(i/2)]=vote_diff_sen[i+1]
        
corr_sen=np.zeros(14)
for i in range(14):
    corr_sen[i]=np.corrcoef(trait_sen_new[:,i], vote_diff_sen_true)[0,1]
 
svc_sen_2layer=svm.LinearSVC(fit_intercept=False,C=param_sen_2layer['C'])
svc_sen_2layer.fit(trait_sen_new, vote_diff_sen_new) 
w_sen=svc_sen_2layer.coef_    
w_sen=w_sen[0]

#governor model
vote_diff_gov_true=np.zeros(int(m/2))
for i in range(m):
    if i%2==0 and i%4==0:
        vote_diff_gov_true[int(i/2)]=vote_diff_gov[i]
    if i%2==0 and i%4==2:
        vote_diff_gov_true[int(i/2)]=vote_diff_gov[i+1]
    
corr_gov=np.zeros(14)
for i in range(14):
    corr_gov[i]=np.corrcoef(trait_gov_new[:,i], vote_diff_gov_true)[0,1]
    
svc_gov_2layer=svm.LinearSVC(fit_intercept=False,C=param_gov_2layer['C'])
svc_gov_2layer.fit(trait_gov_new, vote_diff_gov_new) 
w_gov=svc_gov_2layer.coef_   
w_gov=w_gov[0]

trait_name=["Old","Masculine","Baby-faced","Competent","Attractive","Energetic","Well-groomed","Intelligent","Honest","Generous","Trustworthy","Confident","Rich","Dominant"]
 
#radar plot for corr
df_corr = pd.DataFrame({
'group': ['Senator','Governor'],
trait_name[0]: [corr_sen[0],corr_gov[0]],
trait_name[1]: [corr_sen[1],corr_gov[1]],
trait_name[2]: [corr_sen[2],corr_gov[2]],
trait_name[3]: [corr_sen[3],corr_gov[3]],
trait_name[4]: [corr_sen[4],corr_gov[4]],
trait_name[5]: [corr_sen[5],corr_gov[5]],
trait_name[6]: [corr_sen[6],corr_gov[6]],
trait_name[7]: [corr_sen[7],corr_gov[7]],
trait_name[8]: [corr_sen[8],corr_gov[8]],
trait_name[9]: [corr_sen[9],corr_gov[9]],
trait_name[10]: [corr_sen[10],corr_gov[10]],
trait_name[11]: [corr_sen[11],corr_gov[11]],
trait_name[12]: [corr_sen[12],corr_gov[12]],
trait_name[13]: [corr_sen[13],corr_gov[13]]
})
    
categories=list(df_corr)[:-1]
N = len(categories) 
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure()
ax = plt.subplot(111, polar=True) 
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
plt.yticks([-0.4,-0.2,0,0.2, 0.4], ["-0.4","-0.2","0","0.2","0.4"], color="grey", size=7)
plt.ylim(-0.4,0.4)

values=df_corr.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Senator")
ax.fill(angles, values, 'b', alpha=0.1)
 
values=df_corr.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Governor")
ax.fill(angles, values, 'r', alpha=0.1)
 
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Correlation Coefficient')
plt.savefig('Radar Plot for Corr')

#radar plot for w
df_w = pd.DataFrame({
'group': ['Senator','Governor'],
trait_name[0]: [w_sen[0],w_gov[0]],
trait_name[1]: [w_sen[1],w_gov[1]],
trait_name[2]: [w_sen[2],w_gov[2]],
trait_name[3]: [w_sen[3],w_gov[3]],
trait_name[4]: [w_sen[4],w_gov[4]],
trait_name[5]: [w_sen[5],w_gov[5]],
trait_name[6]: [w_sen[6],w_gov[6]],
trait_name[7]: [w_sen[7],w_gov[7]],
trait_name[8]: [w_sen[8],w_gov[8]],
trait_name[9]: [w_sen[9],w_gov[9]],
trait_name[10]: [w_sen[10],w_gov[10]],
trait_name[11]: [w_sen[11],w_gov[11]],
trait_name[12]: [w_sen[12],w_gov[12]],
trait_name[13]: [w_sen[13],w_gov[13]]
})
    
categories=list(df_w)[:-1]
N = len(categories) 
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure() 
ax = plt.subplot(111, polar=True) 
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
plt.yticks([-10,-5,0,5,10], ["-10","-5","0","5","10"], color="grey", size=7)
plt.ylim(-10,10)

values=df_w.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Senator")
ax.fill(angles, values, 'b', alpha=0.1)
 
values=df_w.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Governor")
ax.fill(angles, values, 'r', alpha=0.1)
 
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Prediction Coefficient')
plt.savefig('Radar Plot for Coefficient')