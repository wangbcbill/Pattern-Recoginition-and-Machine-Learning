from sklearn import svm
import numpy as np  
from function import svc_param_selection, svc_acc

#2.2
trait_sen=np.zeros(n*14).reshape(n,14)
trait_gov=np.zeros(m*14).reshape(m,14)
for i in range(14):
    svr=svm.SVR(kernel='rbf',C=param_rich[i]['C'],gamma=param_rich[i]['gamma'],epsilon=param_rich[i]['epsilon'])
    svr.fit(feature_rich, trait_anno[:,i])
    trait_sen[:,i]=svr.predict(feature_rich_sen)
    trait_gov[:,i]=svr.predict(feature_rich_gov)

#senator model
trait_sen_new=np.zeros(int(n*trait_sen.shape[1]/2)).reshape(int(n/2),trait_sen.shape[1])
for i in range(n):
    if i%2==0 and i%4==0:
        trait_sen_new[int(i/2),:]=trait_sen[i,:]-trait_sen[i+1,:]
    elif i%2==0 and i%4==2:
        trait_sen_new[int(i/2),:]=trait_sen[i+1,:]-trait_sen[i,:]
        
param_sen_2layer=svc_param_selection(trait_sen_new, vote_diff_sen_new, 5) 
acc_sen_2layer=svc_acc(trait_sen_new, vote_diff_sen_new, 5, param_sen_2layer) 
     
#governor model
trait_gov_new=np.zeros(int(m*trait_gov.shape[1]/2)).reshape(int(m/2),trait_gov.shape[1])
for i in range(m):
    if i%2==0 and i%4==0:
        trait_gov_new[int(i/2),:]=trait_gov[i,:]-trait_gov[i+1,:]
    elif i%2==0 and i%4==2:
        trait_gov_new[int(i/2),:]=trait_gov[i+1,:]-trait_gov[i,:]

param_gov_2layer=svc_param_selection(trait_gov_new, vote_diff_gov_new, 5) 
acc_gov_2layer=svc_acc(trait_gov_new, vote_diff_gov_new, 5, param_gov_2layer) 