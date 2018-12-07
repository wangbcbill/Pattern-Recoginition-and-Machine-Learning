from tqdm import tqdm
import numpy as np  
import matplotlib.pyplot as plt  
from function import svr_param_selection, svr_mse_acc, hog_feature
import cv2
import glob

#1.2
images=np.array([cv2.imread(file) for file in glob.glob('./img/*jpg')])

feature_rich=np.concatenate((face_landmark,hog_feature(images)),axis=1)

param_rich=[]
for i in tqdm(range(14)):
    param_rich.append(svr_param_selection(feature_rich, trait_anno[:,i], 5))
    
mseacc_mat_rich=np.zeros(4*14).reshape(14,4)
for i in range(14):
    mseacc_mat_rich[i,:]=svr_mse_acc(feature_rich, trait_anno[:,i], 5, param_rich[i])
    
plt.figure()
plt.plot(np.arange(1,15),mseacc_mat[:,0], label = 'train mse poor')
plt.plot(np.arange(1,15),mseacc_mat[:,1], label = 'test mse poor')
plt.plot(np.arange(1,15),mseacc_mat_rich[:,0], label = 'train mse rich')
plt.plot(np.arange(1,15),mseacc_mat_rich[:,1], label = 'test mse rich')
plt.legend(loc = 'upper right')
plt.title('Mean Squared Error using richer features')
plt.ylabel('Mean Squared Error')
plt.xlabel('Facial traits')
plt.savefig('MSE using richer features')

plt.figure()
plt.plot(np.arange(1,15),mseacc_mat[:,2], label = 'train acc poor')
plt.plot(np.arange(1,15),mseacc_mat[:,3], label = 'test acc poor')
plt.plot(np.arange(1,15),mseacc_mat_rich[:,2], label = 'train acc rich')
plt.plot(np.arange(1,15),mseacc_mat_rich[:,3], label = 'test acc rich')
plt.legend(loc = 'upper right')
plt.title('Classification Accuracy using richer features')
plt.ylabel('Classification Accuracy')
plt.xlabel('Facial traits')
plt.savefig('ACC using richer features')
