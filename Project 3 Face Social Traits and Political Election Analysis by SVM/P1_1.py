from tqdm import tqdm
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.io import loadmat
import os
os.chdir("C:/backup/2018Fall/231/project4_code_and_data")
from function import scale, svr_param_selection, svr_mse_acc

#1.1 traing only using landmarks
train=loadmat("./train-anno.mat")
face_landmark=train['face_landmark']
trait_anno=train['trait_annotation']

for i in range(face_landmark.shape[1]):
    face_landmark[:,i]=scale(face_landmark[:,i])

param=[]
for i in tqdm(range(14)):
    param.append(svr_param_selection(face_landmark, trait_anno[:,i], 5))

mseacc_mat=np.zeros(4*14).reshape(14,4)
for i in range(14):
    mseacc_mat[i,:]=svr_mse_acc(face_landmark, trait_anno[:,i], 5, param[i])

plt.figure()
plt.plot(np.arange(1,15),mseacc_mat[:,0], label = 'train mse')
plt.plot(np.arange(1,15),mseacc_mat[:,1], label = 'test mse')
plt.legend(loc = 'upper right')
plt.title('Mean Squared Error only using landmarks')
plt.ylabel('Mean Squared Error')
plt.xlabel('Facial traits')
plt.savefig('MSE only using landmarks')

plt.figure()
plt.plot(np.arange(1,15),mseacc_mat[:,2], label = 'train acc')
plt.plot(np.arange(1,15),mseacc_mat[:,3], label = 'test acc')
plt.legend(loc = 'upper right')
plt.title('Classification Accuracy only using landmarks')
plt.ylabel('Classification Accuracy')
plt.xlabel('Facial traits')
plt.savefig('ACC only using landmarks')
