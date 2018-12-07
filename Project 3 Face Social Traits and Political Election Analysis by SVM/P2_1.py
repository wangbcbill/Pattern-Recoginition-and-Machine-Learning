import numpy as np  
from scipy.io import loadmat
from function import scale, svc_param_selection, svc_acc, hog_feature
import cv2
import glob

#2.1 
#senator model
stat_sen=loadmat("./stat-sen.mat")
face_landmark_sen=stat_sen['face_landmark']
vote_diff_sen=stat_sen['vote_diff']

for i in range(face_landmark_sen.shape[1]):
    face_landmark_sen[:,i]=scale(face_landmark_sen[:,i])

images_sen=np.array([cv2.imread(file) for file in glob.glob('./img-elec/senator/*jpg')])
feature_rich_sen=np.concatenate((face_landmark_sen,hog_feature(images_sen)),axis=1)

n=images_sen.shape[0]
feature_rich_sen_new=np.zeros(int(n*feature_rich_sen.shape[1]/2)).reshape(int(n/2),feature_rich_sen.shape[1])
vote_diff_sen_new=np.zeros(int(n/2))
for i in range(n):
    if i%2==0 and i%4==0:
        feature_rich_sen_new[int(i/2),:]=feature_rich_sen[i,:]-feature_rich_sen[i+1,:]
        if vote_diff_sen[i]-vote_diff_sen[i+1]>0:
            vote_diff_sen_new[int(i/2)]=1
        else:
            vote_diff_sen_new[int(i/2)]=-1
    elif i%2==0 and i%4==2:
        feature_rich_sen_new[int(i/2),:]=feature_rich_sen[i+1,:]-feature_rich_sen[i,:]
        if vote_diff_sen[i+1]-vote_diff_sen[i]>0:
            vote_diff_sen_new[int(i/2)]=1
        else:
            vote_diff_sen_new[int(i/2)]=-1
  
param_sen=svc_param_selection(feature_rich_sen_new, vote_diff_sen_new, 5)
acc_sen=svc_acc(feature_rich_sen_new, vote_diff_sen_new, 5, param_sen)

#governor model
stat_gov=loadmat("./stat-gov.mat")
face_landmark_gov=stat_gov['face_landmark']
vote_diff_gov=stat_gov['vote_diff']

for i in range(face_landmark_gov.shape[1]):
    face_landmark_gov[:,i]=scale(face_landmark_gov[:,i])

images_gov=np.array([cv2.imread(file) for file in glob.glob('./img-elec/governor/*jpg')])
feature_rich_gov=np.concatenate((face_landmark_gov,hog_feature(images_gov)),axis=1)

m=images_gov.shape[0]
feature_rich_gov_new=np.zeros(int(m*feature_rich_gov.shape[1]/2)).reshape(int(m/2),feature_rich_gov.shape[1])
vote_diff_gov_new=np.zeros(int(m/2))
for i in range(m):
    if i%2==0 and i%4==0:
        feature_rich_gov_new[int(i/2),:]=feature_rich_gov[i,:]-feature_rich_gov[i+1,:]
        if vote_diff_gov[i]-vote_diff_gov[i+1]>0:
            vote_diff_gov_new[int(i/2)]=1
        else:
            vote_diff_gov_new[int(i/2)]=-1
    elif i%2==0 and i%4==2:
        feature_rich_gov_new[int(i/2),:]=feature_rich_gov[i+1,:]-feature_rich_gov[i,:]
        if vote_diff_gov[i+1]-vote_diff_gov[i]>0:
            vote_diff_gov_new[int(i/2)]=1
        else:
            vote_diff_gov_new[int(i/2)]=-1

param_gov=svc_param_selection(feature_rich_gov_new, vote_diff_gov_new, 5)   
acc_gov=svc_acc(feature_rich_gov_new, vote_diff_gov_new, 5, param_gov)  

