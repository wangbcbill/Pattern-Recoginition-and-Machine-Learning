import numpy as np  
from sklearn import svm, model_selection, metrics
from skimage.feature import hog

def scale(X):
    imin=X.min()
    imax=X.max()
    if imin!=imax:
        X_scale=(X-imin)/(imax-imin)
    else:
        X_scale=X/X
    return X_scale

def svr_param_selection(X, y, nfolds):
    Cs = [0.01, 0.1, 1, 10, 100, 500, 1000, 5000]
    gammas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    epsilons = [0.0001, 0.001, 0.01, 0.1]
    param_grid = {'C': Cs, 'gamma' : gammas, 'epsilon' : epsilons}
    grid_search = model_selection.GridSearchCV(svm.SVR(kernel='rbf'), param_grid,scoring='neg_mean_squared_error',n_jobs=-1, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def svr_mse_acc(X, y, nfolds, param):
    mse_train=[]
    mse_test=[]
    acc_train=[]
    acc_test=[]
    kf = model_selection.KFold(n_splits=nfolds)
    svr=svm.SVR(kernel='rbf',C=param['C'],gamma=param['gamma'],epsilon=param['epsilon'])
    threshold=np.mean(y)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svr.fit(X_train,y_train)
        y_train_pred=svr.predict(X_train)
        y_test_pred=svr.predict(X_test)
        mse_train.append(metrics.mean_squared_error(y_train, y_train_pred))
        mse_test.append(metrics.mean_squared_error(y_test, y_test_pred))
        acc_train.append(np.mean((y_train>threshold)*(y_train_pred>threshold))+np.mean((y_train<=threshold)*(y_train_pred<=threshold)))
        acc_test.append(np.mean((y_test>threshold)*(y_test_pred>threshold))+np.mean((y_test<=threshold)*(y_test_pred<=threshold)))
    return np.mean(mse_train), np.mean(mse_test), np.mean(acc_train), np.mean(acc_test)

def hog_feature(images):
    hog_feature=[]
    for i in range(images.shape[0]):
        hog_feature.append(hog(images[i,:,:,:],orientations=32,pixels_per_cell=(35,35),cells_per_block=(1,1),feature_vector=True))    
    hog_feature_mat=np.zeros(len(hog_feature)*hog_feature[0].shape[0]).reshape(len(hog_feature),hog_feature[0].shape[0]) 
    for i in range(images.shape[0]):
        hog_feature_mat[i,:]=hog_feature[i]
    return hog_feature_mat

def svc_param_selection(X, y, nfolds):
    Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 5000, 10000]
    param_grid = {'C': Cs}
    grid_search = model_selection.GridSearchCV(svm.LinearSVC(fit_intercept=False), param_grid, n_jobs=-1,cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def svc_acc(X, y, nfolds, param):
    acc_train=[]
    acc_test=[]
    kf = model_selection.KFold(n_splits=nfolds)
    svc=svm.LinearSVC(fit_intercept=False,C=param['C'])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svc.fit(X_train,y_train)
        y_train_pred=svc.predict(X_train)
        y_test_pred=svc.predict(X_test)
        acc_train.append(np.mean(y_train==y_train_pred))
        acc_test.append(np.mean(y_test==y_test_pred))
    return np.mean(acc_train), np.mean(acc_test)

