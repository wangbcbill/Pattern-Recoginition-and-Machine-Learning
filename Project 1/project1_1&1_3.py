# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:26:24 2018

@author: dell
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import datetime
import imageio
import glob
from skimage import color

##################################
# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True

#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        p1=(int(p[0]),int(p[1]))
        if p1[1]<=rect[2]-1 and p1[0]<=rect[2]-1 and p1[1]>=rect[0] and p1[0]>=rect[0]:
            subdiv.insert(p1) 
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
        
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    w,h,num_chans = img1.shape
    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], num_chans), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    if num_chans==1:
        img2Rect=np.reshape(img2Rect,(r2[3], r2[2], num_chans))
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    if num_chans==1:
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( 1.0 - mask )
     
    else:
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 

###################################
def warp(Image,sc,tc):
    '''
    Image: the image to be warped
    sc: original landmarks
    tc: warped landmarks
    '''
    HW,_,_=Image.shape
    cornerps=[[0,0],[0,HW-1],[HW-1,0],[HW-1,HW-1]]
    #cornerps=[[0,0],[0,HW-1],[HW-1,0],[HW-1,HW-1],[0,np.floor(HW/2)],[np.floor(HW/2),0],[HW-1,np.floor(HW/2)],[np.floor(HW/2),HW-1]]

    scl=sc.astype(np.int64).tolist()+cornerps
    tcl=tc.astype(np.int64).tolist()+cornerps
    imgWarped = np.copy(Image);    
    rect = (0, 0, HW, HW)
    dt = calculateDelaunayTriangles(rect,tcl)
# Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        
        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(scl[dt[i][j]])
            t2.append(tcl[dt[i][j]])
        
        warpTriangle(Image, imgWarped, t1, t2)
    return imgWarped

#########################################
def plot(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(Nc, Nh))
    plt.clf()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[0:Nh*Nc,:,:,:]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
            immin=(image[:,:]).min()
            immax=(image[:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image,cmap ='gray')
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
            immin=(image[:,:,:]).min()
            immax=(image[:,:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image)
#########################################

images=np.array([cv2.imread(file) for file in glob.glob('C:/backup/2018Fall/231/Project_1/Project_1/images/*jpg')])

image_hsv=np.zeros([1000,128,128,3])
for i in range(1000):
    image_hsv[i,:,:,:]=color.rgb2hsv(cv2.cvtColor(images[i,:,:,:], cv2.COLOR_BGR2RGB))

#2.1.1
np.random.seed(231)
#from sklearn.model_selection import train_test_split
#indices=np.arange(1000)
#train_hsv, test_hsv,train_ind,test_ind = train_test_split(image_hsv, indices,test_size=0.2)
train_hsv=image_hsv[0:800,:,:,:]
test_hsv=image_hsv[800:1000,:,:,:]
train_ind=np.arange(800)
test_ind=np.arange(800,1000)
train=train_hsv[:,:,:,2]
test=test_hsv[:,:,:,2]
train=np.reshape(train, (train.shape[0], -1))
test=np.reshape(test, (test.shape[0], -1))

from sklearn import decomposition
pca = decomposition.PCA(n_components=50, whiten=True)
pca.fit(train)

plt.imshow(pca.mean_.reshape([128,128]),cmap=plt.cm.bone)

fig = plt.figure(figsize=(16, 6))
for i in range(10):
    ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape([128,128]),cmap=plt.cm.bone)

f=np.zeros([200,16384])
fig = plt.figure(figsize=(16, 6))
for i in range(200):
    f[i,:] = pca.mean_
    for j in range(50):
        f[i,:] = f[i,:] + np.dot(test[i]-pca.mean_,pca.components_[j])*pca.components_[j]
    a=color.hsv2rgb(cv2.merge([test_hsv[i,:,:,0], test_hsv[i,:,:,1], f[i,:].reshape([128,128])]))
    amin=a.min()
    amax=a.max()
    a=(a-amin)/(amax-amin+1e-8)
    ax = fig.add_subplot(10, 20, i + 1, xticks=[], yticks=[])
    ax.imshow(a,cmap=plt.cm.bone)

f=np.zeros([200,16384])
fig = plt.figure(figsize=(16, 6))
for i in range(10):
    f[i,:] = pca.mean_
    for j in range(50):
        f[i,:] = f[i,:] + np.dot(test[i]-pca.mean_,pca.components_[j])*pca.components_[j]
    a=color.hsv2rgb(cv2.merge([test_hsv[i,:,:,0], test_hsv[i,:,:,1], f[i,:].reshape([128,128])]))
    amin=a.min()
    amax=a.max()
    a=(a-amin)/(amax-amin+1e-8)
    ax = fig.add_subplot(1,10, i + 1, xticks=[], yticks=[])
    ax.imshow(a,cmap=plt.cm.bone)
    
    
fig = plt.figure(figsize=(16, 6))
for i in range(200):
    a=color.hsv2rgb(test_hsv[i,:,:,:])
    amin=a.min()
    amax=a.max()
    a=(a-amin)/(amax-amin+1e-8)
    ax = fig.add_subplot(10, 20, i + 1, xticks=[], yticks=[])
    ax.imshow(a,cmap=plt.cm.bone)

fig = plt.figure(figsize=(16, 6))
for i in range(10):
    a=color.hsv2rgb(test_hsv[i,:,:,:])
    amin=a.min()
    amax=a.max()
    a=(a-amin)/(amax-amin+1e-8)
    ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(a,cmap=plt.cm.bone)

error=np.zeros(11)
a=np.array([1,5,10,15,20,25,30,35,40,45,50])
f=np.zeros([200,16384])
for t in range(11):
    pca1 = decomposition.PCA(n_components=a[t], whiten=True)
    pca1.fit(train)
    for i in range(200):
        f[i,:] = pca1.mean_
        for j in range(a[t]):
            f[i,:] = f[i,:] + np.dot(test[i]-pca1.mean_,pca1.components_[j])*pca1.components_[j]
        error[t]=error[t]+sum((f[i,:]-test[i])**2)
    error[t]=error[t]/(128*128*200)

plt.plot(a,error)
plt.ylabel("Reconstruct Error")
plt.xlabel("N_components")


#error_rgb=np.zeros(50)
#f=np.zeros([200,16384])
#for t in range(1,51):
#    pca1 = decomposition.PCA(n_components=t, whiten=True)
#    pca1.fit(train)
#    for i in range(200):
#        f[i,:] = pca1.mean_
#        for j in range(t):
#            f[i,:] = f[i,:] + np.dot(test[i]-pca1.mean_,pca1.components_[j])*pca1.components_[j]
#        b=(color.hsv2rgb(test_hsv[i,:,:,:])-color.hsv2rgb(cv2.merge([test_hsv[i,:,:,0], test_hsv[i,:,:,1], f[i,:].reshape([128,128])])))**2
#        error_rgb[t-1]=error_rgb[t-1]+b.sum()
#    error_rgb[t-1]=error_rgb[t-1]/(128*128*200)
 
#plt.plot(np.arange(1,51),error_rgb)
#plt.ylabel("Reconstruct Error")
#plt.xlabel("N_components")

#2.1.2
import scipy.io
landmarks = np.array([scipy.io.loadmat(file)['lms'] for file in glob.glob('C:/backup/2018Fall/231/Project_1/Project_1/landmarks/*mat')])

train_l=landmarks[train_ind]
test_l=landmarks[test_ind]
train_l=np.reshape(train_l, (train_l.shape[0], -1))
test_l=np.reshape(test_l, (test_l.shape[0], -1))

pca_l = decomposition.PCA(n_components=50, whiten=True)
pca_l.fit(train_l)

plt.figure(figsize=(7, 6))
plt.scatter(pca_l.mean_.reshape([68,2])[:,0],pca_l.mean_.reshape([68,2])[:,1])
plt.xlim(0, 128)
plt.ylim(128, 0)

plt.figure(figsize=(7, 6))
for i in range(10):
    scatter=(pca_l.components_[i]+pca_l.mean_).reshape([68,2])
    plt.scatter(scatter[:,0],scatter[:,1])
    plt.xlim(0, 128)
    plt.ylim(128, 0)

error_l=np.zeros(11)
a=np.array([1,5,10,15,20,25,30,35,40,45,50])
f=np.zeros([200,68*2])
for t in range(11):
    pca2 = decomposition.PCA(n_components=a[t], whiten=True)
    pca2.fit(train_l)
    for i in range(200):
        f[i,:] = pca2.mean_
        for j in range(a[t]):
            f[i,:] = f[i,:] + np.dot(test_l[i]-pca2.mean_,pca2.components_[j])*pca2.components_[j]
        b=(test_l[i]-f[i,:])**2
        error_l[t]=error_l[t]+np.sqrt(b.sum())
    error_l[t]=error_l[t]/200
 
plt.plot(a,error_l)
plt.ylabel("Reconstruct Error")
plt.xlabel("N_components")
    
#2.1.3
train_w=np.zeros([800,128,128,3])
for i in range(800):
    train_w[i,:,:,:]=warp(train_hsv[i,:,:,:],landmarks[train_ind][i,:,:],pca_l.mean_.reshape([68,2]))

train_w=train_w[:,:,:,2]
train_w=np.reshape(train_w, (train_w.shape[0], -1))
pca_w = decomposition.PCA(n_components=50, whiten=True)
pca_w.fit(train_w)

test_w=np.zeros([200,128,128,3])
test_w1=np.zeros([200,128*128])
test_re_l=np.zeros([200,136])
test_re=np.zeros([200,16384])
test_re_w=np.zeros([200,128,128,1])
fig = plt.figure(figsize=(16, 6))
img=np.zeros([200,128,128,3])
for i in range(200):
    test_re_l[i,:] = pca_l.mean_
    for j in range(10):
        test_re_l[i,:] = test_re_l[i,:] + np.dot(test_l[i]-pca_l.mean_,pca_l.components_[j])*pca_l.components_[j]
    test_w[i,:,:,:]=warp(test_hsv[i,:,:,:],landmarks[test_ind][i,:,:],pca_l.mean_.reshape([68,2]))
    test_w1[i,:]=np.reshape(test_w[i,:,:,2],(128*128))
    test_re[i,:] = pca_w.mean_ 
    for t in range(50):
        test_re[i,:] = test_re[i,:] + np.dot(test_w1[i]-pca_w.mean_,pca_w.components_[t])*pca_w.components_[t]  
    test_re_w[i,:,:,:]=warp(test_re[i,:].reshape([128,128,1]),pca_l.mean_.reshape([68,2]), test_re_l[i,:].reshape([68,2]))
    a=color.hsv2rgb(cv2.merge([test_hsv[i,:,:,0], test_hsv[i,:,:,1], test_re_w[i,:].reshape([128,128])]))
    img[i]=np.copy(a)
    amin=a.min()
    amax=a.max()
    a=(a-amin)/(amax-amin+1e-8)
    ax = fig.add_subplot(10, 20, i + 1, xticks=[], yticks=[])
    ax.imshow(a,cmap=plt.cm.bone)

plot(img[0:20,:,:,:],2,10,3,128,128)

original=np.zeros([20,128,128,3])
for i in range(20):
    original[i,:,:,:]=color.hsv2rgb(test_hsv[i,:,:,:])

plot(original[:,:,:,:],2,10,3,128,128)

error_w=np.zeros(11)
test_re=np.zeros([200,16384])
test_re_w=np.zeros([200,128,128,1])
a=np.array([1,5,10,15,20,25,30,35,40,45,50])
for n in range(11):
    pca_w3 = decomposition.PCA(n_components=a[n], whiten=True)
    pca_w3.fit(train_w)
    for i in range(200):
        test_re[i,:] = pca_w3.mean_ 
        for t in range(a[n]):
            test_re[i,:] = test_re[i,:] + np.dot(test_w1[i]-pca_w3.mean_,pca_w3.components_[t])*pca_w3.components_[t]  
        test_re_w[i,:,:,:]=warp(test_re[i,:].reshape([128,128,1]),pca_l.mean_.reshape([68,2]), test_re_l[i,:].reshape([68,2]))
        b=(test[i]- test_re_w[i,:,:,:].reshape([128*128]))**2
        error_w[n]=error_w[n]+b.sum()
    error_w[n]=error_w[n]/(200*128*128)

plt.plot(a,error_w)
plt.ylabel("Reconstruct Error")
plt.xlabel("N_components")

#2.1.4
sample_w=np.random.multivariate_normal(np.zeros(50),np.diag(pca.explained_variance_),50)
sample_l_w=np.random.multivariate_normal(np.zeros(10),np.diag(pca_l.explained_variance_[0:10]),50)
sample=np.zeros([50,128*128])
sample_l=np.zeros([50,68*2])
sample_img=np.zeros([50,128,128,1])
fig = plt.figure(figsize=(16, 6))
for i in range(50):
    sample[i,:]=np.dot(sample_w[i],pca.components_[0:50])+pca.mean_
    sample_l[i,:]=np.dot(sample_l_w[i],pca_l.components_[0:10])+pca_l.mean_
    sample_img[i,:,:,:]=warp(sample[i,:].reshape([128,128,1]),pca_l.mean_.reshape([68,2]),sample_l[i,:].reshape([68,2]))
    a=sample_img[i,:,:,:].reshape([128,128])
    amin=a.min()
    amax=a.max()
    a=(a-amin)/(amax-amin+1e-8)
    ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(a,cmap='gray')
    

#2.3.1
images_m=np.array([cv2.imread(file) for file in glob.glob('C:/backup/2018Fall/231/Project_1/Project_1/male_images/*jpg')])
np.save('images_m.npy', images_m)
images_f=np.array([cv2.imread(file) for file in glob.glob('C:/backup/2018Fall/231/Project_1/Project_1/female_images/*jpg')])
np.save('images_f.npy', images_f)

import scipy.io
landmarks_m=np.array([scipy.io.loadmat(file)['lms'] for file in glob.glob('C:/backup/2018Fall/231/Project_1/Project_1/male_landmarks/*mat')])
np.save('landmarks_m.npy', landmarks_m)
landmarks_f=np.array([scipy.io.loadmat(file)['lms'] for file in glob.glob('C:/backup/2018Fall/231/Project_1/Project_1/female_landmarks/*mat')])
np.save('landmarks_f.npy', landmarks_f)

image_m_hsv=np.zeros([412,128,128,3])
for i in range(412):
    image_m_hsv[i,:,:,:]=color.rgb2hsv(cv2.cvtColor(images_m[i,:,:,:], cv2.COLOR_BGR2RGB))

image_f_hsv=np.zeros([588,128,128,3])
for i in range(588):
    image_f_hsv[i,:,:,:]=color.rgb2hsv(cv2.cvtColor(images_f[i,:,:,:], cv2.COLOR_BGR2RGB))

image_fld=np.concatenate((image_m_hsv,image_f_hsv), axis = 0)

np.random.seed(231)
from sklearn.model_selection import train_test_split
indices_fld=np.arange(1000)
train_fld_hsv, test_fld_hsv,train_fld_ind,test_fld_ind = train_test_split(image_fld, indices_fld,test_size=0.2)
train_fld=train_fld_hsv[:,:,:,2]
test_fld=test_fld_hsv[:,:,:,2]
train_fld=np.reshape(train_fld, (train_fld.shape[0], -1))
test_fld=np.reshape(test_fld, (test_fld.shape[0], -1))

from sklearn import decomposition
pca_fld = decomposition.PCA(n_components=50, whiten=True)
pca_fld.fit(train_fld)

x=np.zeros([800,50])
for i in range(800):
    for j in range(50):
        x[i,j] = np.dot(train_fld[i]-pca_fld.mean_,pca_fld.components_[j])
        
male_ind=np.zeros(800)
for i in range(800):
    if train_fld_ind[i]>411:
        male_ind[i]=0
    else:
        male_ind[i]=1

sum_m=np.zeros([50,1])
sum_f=np.zeros([50,1])    
for i in range(800):
    if male_ind[i]==1:
        sum_m+=x[i,:].reshape([50,1])
    else:
        sum_f+=x[i,:].reshape([50,1])

mean_m=sum_m/male_ind.sum()
mean_f=sum_f/(800-male_ind.sum())

s_m=np.zeros([50,50])
s_f=np.zeros([50,50])
for i in range(800):
    if male_ind[i]==1:
        s_m+=np.dot(x[i,:].reshape([50,1])-mean_m,(x[i,:].reshape([50,1])-mean_m).T)
    else:
        s_f+=np.dot(x[i,:].reshape([50,1])-mean_f,(x[i,:].reshape([50,1])-mean_f).T)

w=np.dot(np.matrix(s_m+s_f).I,(mean_m-mean_f))

FisherFace=pca_fld.mean_
for i in range(50):
   FisherFace += w[i,0]*pca_fld.components_[i]

FisherFace0=np.zeros([128*128])
for i in range(50):
   FisherFace0 += w[i,0]*pca_fld.components_[i]

plt.imshow(FisherFace.reshape([128,128]),cmap=plt.cm.bone)

plot(FisherFace0.reshape([1,128,128,1]),1,1,1,128,128)

y=np.zeros([800])
for i in range(800):
    y[i]=np.dot(w.T,x[i,:].reshape([50,1]))[0,0]

w0_1=np.dot(w.T,mean_m+mean_f)/2
w0_1=w0_1[0,0]
#0.0005808396772934234

w0_2=(male_ind.sum()*np.dot(w.T,mean_m)[0,0]+(800-male_ind.sum())*np.dot(w.T,mean_f)[0,0])/800

error1=0
for i in range(800):
    if y[i]>=w0_1:
        if male_ind[i]==0:
            error1+=1
    if y[i]<w0_1:
        if male_ind[i]==1:
            error1+=1

error2=0
for i in range(800):
    if y[i]>=w0_2:
        if male_ind[i]==0:
            error2+=1
    if y[i]<w0_2:
        if male_ind[i]==1:
            error2+=1

#prefer w0_1
x_test=np.zeros([200,50])
for i in range(200):
    for j in range(50):
        x_test[i,j] = np.dot(test_fld[i]-pca_fld.mean_,pca_fld.components_[j])

male_ind_test=np.zeros(200)
for i in range(200):
    if test_fld_ind[i]>411:
        male_ind_test[i]=0
    else:
        male_ind_test[i]=1

y_test=np.zeros([200])
for i in range(200):
    y_test[i]=np.dot(w.T,x_test[i,:].reshape([50,1]))[0,0]

error_test=0
for i in range(200):
    if y_test[i]>=w0_1:
        if male_ind_test[i]==0:
            error_test+=1
    if y_test[i]<w0_1:
        if male_ind_test[i]==1:
            error_test+=1

#error rate 0.125
error_test/200

#2.3.2
##landmarks
landmarks_fld=np.concatenate((landmarks_m,landmarks_f), axis = 0)

train_fld_l=np.reshape(landmarks_fld[train_fld_ind], (landmarks_fld[train_fld_ind].shape[0], -1))
test_fld_l=np.reshape(landmarks_fld[test_fld_ind], (landmarks_fld[test_fld_ind].shape[0], -1))

pca_fld_l = decomposition.PCA(n_components=10, whiten=True)
pca_fld_l.fit(train_fld_l)

x_l=np.zeros([800,10])
for i in range(800):
    for j in range(10):
        x_l[i,j] = np.dot(train_fld_l[i]-pca_fld_l.mean_,pca_fld_l.components_[j])

sum_m_l=np.zeros([10,1])
sum_f_l=np.zeros([10,1])    
for i in range(800):
    if male_ind[i]==1:
        sum_m_l+=x_l[i,:].reshape([10,1])
    else:
        sum_f_l+=x_l[i,:].reshape([10,1])

mean_m_l=sum_m_l/male_ind.sum()
mean_f_l=sum_f_l/(800-male_ind.sum())

s_m_l=np.zeros([10,10])
s_f_l=np.zeros([10,10])
for i in range(800):
    if male_ind[i]==1:
        s_m_l+=np.dot(x_l[i,:].reshape([10,1])-mean_m_l,(x_l[i,:].reshape([10,1])-mean_m_l).T)
    else:
        s_f_l+=np.dot(x_l[i,:].reshape([10,1])-mean_f_l,(x_l[i,:].reshape([10,1])-mean_f_l).T)

w_l=np.dot(np.matrix(s_m_l+s_f_l).I,(mean_m_l-mean_f_l))

y_l=np.zeros([800])
for i in range(800):
    y_l[i]=np.dot(w_l.T,x_l[i,:].reshape([10,1]))[0,0]

x_test_l=np.zeros([200,10])
for i in range(200):
    for j in range(10):
        x_test_l[i,j] = np.dot(test_fld_l[i]-pca_fld_l.mean_,pca_fld_l.components_[j])

y_test_l=np.zeros([200])
for i in range(200):
    y_test_l[i]=np.dot(w_l.T,x_test_l[i,:].reshape([10,1]))[0,0]

##aligned face
train_align=np.zeros([800,128,128,3])
for i in range(800):
    train_align[i,:,:,:]=warp(train_fld_hsv[i,:,:,:],landmarks_fld[train_fld_ind][i,:,:],pca_fld_l.mean_.reshape([68,2]))

train_align=train_align[:,:,:,2]
train_align=np.reshape(train_align, (train_align.shape[0], -1))
pca_fld_a = decomposition.PCA(n_components=50, whiten=True)
pca_fld_a.fit(train_align)

test_align=np.zeros([200,128,128,3])
for i in range(200):
    test_align[i,:,:,:]=warp(test_fld_hsv[i,:,:,:],landmarks_fld[test_fld_ind][i,:,:],pca_fld_l.mean_.reshape([68,2]))

test_align=test_align[:,:,:,2]
test_align=np.reshape(test_align, (test_align.shape[0], -1))

x_a=np.zeros([800,50])
for i in range(800):
    for j in range(50):
        x_a[i,j] = np.dot(train_align[i]-pca_fld_a.mean_,pca_fld_a.components_[j])

sum_m_a=np.zeros([50,1])
sum_f_a=np.zeros([50,1])    
for i in range(800):
    if male_ind[i]==1:
        sum_m_a+=x_a[i,:].reshape([50,1])
    else:
        sum_f_a+=x_a[i,:].reshape([50,1])

mean_m_a=sum_m_a/male_ind.sum()
mean_f_a=sum_f_a/(800-male_ind.sum())

s_m_a=np.zeros([50,50])
s_f_a=np.zeros([50,50])
for i in range(800):
    if male_ind[i]==1:
        s_m_a+=np.dot(x_a[i,:].reshape([50,1])-mean_m_a,(x_a[i,:].reshape([50,1])-mean_m_a).T)
    else:
        s_f_a+=np.dot(x_a[i,:].reshape([50,1])-mean_f_a,(x_a[i,:].reshape([50,1])-mean_f_a).T)

w_a=np.dot(np.matrix(s_m_a+s_f_a).I,(mean_m_a-mean_f_a))

y_a=np.zeros([800])
for i in range(800):
    y_a[i]=np.dot(w_a.T,x_a[i,:].reshape([50,1]))[0,0]

x_test_a=np.zeros([200,50])
for i in range(200):
    for j in range(50):
        x_test_a[i,j] = np.dot(test_align[i]-pca_fld_a.mean_,pca_fld_a.components_[j])

y_test_a=np.zeros([200])
for i in range(200):
    y_test_a[i]=np.dot(w_a.T,x_test_a[i,:].reshape([50,1]))[0,0]

y_test_a_m=y_test_a[male_ind_test==1]
y_test_a_f=y_test_a[male_ind_test==0]
y_test_l_m=y_test_a[male_ind_test==1]
y_test_l_f=y_test_a[male_ind_test==0]
y_a_m=y_a[male_ind==1]
y_a_f=y_a[male_ind==0]
y_l_m=y_l[male_ind==1]
y_l_f=y_l[male_ind==0]
plt.scatter(y_test_a_m,y_test_a_m,c='blue',marker='x',label='test male')
plt.scatter(y_test_a_f,y_test_a_f,c='red',marker='x',label='test female')
plt.scatter(y_a_m,y_l_m,c="blue",marker='o',label='train male')
plt.scatter(y_a_f,y_l_f,c="red",marker='o',label='train female')
plt.xlim(-0.025, 0.02)
plt.ylim(-0.0055, 0.0055)
plt.xlabel("geometric")
plt.ylabel("appearance")
plt.legend(loc='upper left')





