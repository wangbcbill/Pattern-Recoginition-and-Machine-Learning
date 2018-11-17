import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import copy

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize


class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = None      
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
	
	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d images.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations
	
	#select weak classifiers to form a strong classifier
	#after training, by calling self.sc_function(), a prediction can be made
	#self.chosen_wcs should be assigned a value after self.train() finishes
	#call Weak_Classifier.calc_error() in this function
	#cache training results to self.visualizer for visualization
	#
	#
	#detailed implementation is up to you
	#consider caching partial results and using parallel computing
	def train(self, save_dir = None, load_dir = None):
		######################
		######## TODO ########
		######################            

		if self.style=='Ada':
			if load_dir is not None and os.path.exists(load_dir):
				print('[Find cached chosen classifiers, %s loading...]' % load_dir)
				with open(load_dir,'rb') as f:
					chosen_wcs=pickle.load(f) 
				self.chosen_wcs=chosen_wcs		
				for t in range(self.num_chosen_wc):
					h = np.zeros(len(self.data))
					wcs=chosen_wcs[t][1]
					alpha=chosen_wcs[t][0]
					activ=np.array(wcs.activations)
					h[activ>=wcs.threshold],h[activ<wcs.threshold]=wcs.polarity,-wcs.polarity
					if t==0:
					   self.visualizer.strong_classifier_scores[t]=alpha*h   
					else:
					   self.visualizer.strong_classifier_scores[t]=self.visualizer.strong_classifier_scores[t-1]+alpha*h                    
			else:
				m=len(self.data)
				D=np.repeat(1/m,m)
				chosen_wcs=[]
				h = np.zeros(m)
				for t in range(self.num_chosen_wc):
					err=np.array([wc.calc_error(D,self.labels) for wc in self.weak_classifiers])  
					if t+1 in self.visualizer.top_wc_intervals:
					   self.visualizer.weak_classifier_accuracies[t+1]=np.sort(err)[:1000]
					min_idx=np.argmin(err)
					wcs=copy.deepcopy(self.weak_classifiers[min_idx])
					alpha=0.5*np.log((1-np.min(err))/np.min(err))
					activ=np.array(wcs.activations)
					h[activ>=wcs.threshold],h[activ<wcs.threshold]=wcs.polarity,-wcs.polarity
					if t==0:
					   self.visualizer.strong_classifier_scores[t]=alpha*h   
					else:
					   self.visualizer.strong_classifier_scores[t]=self.visualizer.strong_classifier_scores[t-1]+alpha*h
					chosen_wcs.append([alpha,wcs])  
					D_1=D*np.exp(-self.labels*alpha*h)
					D=D_1/sum(D_1)  
					print('In epochs %d, Training Error is %f' %(t+1,np.min(err)))
				self.chosen_wcs=chosen_wcs
			self.visualizer.Haar=chosen_wcs
			self.visualizer.Haar.sort(key = lambda x: x[0], reverse=True)
         

		if save_dir is not None:
			pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))       
		
	def ada2real(self, visualizer):
		m=len(self.data)
		labels=self.labels
		D=np.repeat(1/m,m)
		for t in range(self.num_chosen_wc):
			wcs=self.chosen_wcs[t][1]
			idx=np.argsort(wcs.activations)
			hist,_=np.histogram(wcs.activations,self.num_bins)
			h_val=np.zeros_like(D)
			D_1=np.zeros_like(D)
			for b in range(self.num_bins):
				if b==0:
					i=0    
				j=hist[b]
				selected_idx=idx[i:i+j]
				D_c=D[selected_idx]
				labels_c=labels[selected_idx]   
				p=np.sum(D_c[np.asarray(labels_c)==1])+1e-5              
				q=np.sum(D_c[np.asarray(labels_c)==-1])+1e-5
				h=0.5*np.log(p/q)
				h_val[selected_idx]=h
				i=i+j 
			if t==0:
				visualizer.strong_classifier_scores[t]=h_val  
			else:
				visualizer.strong_classifier_scores[t]=visualizer.strong_classifier_scores[t-1]+h_val
			D_1=D*np.exp(-labels*h_val)	
			D=D_1/sum(D_1)
	
	def sc_function(self, image):
		return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])			

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, name, scale_step = 20):
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		train_predicts = []
		for idx in range(self.data.shape[0]):
			train_predicts.append(self.sc_function(self.data[idx, ...]))
		print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################

		scales = 1 / np.linspace(1, 8, scale_step)
		if os.path.exists('patches_%s.npy' %name):
			patches = np.load('patches_%s.npy' %name)
			patch_xyxy = np.load('patch_position_%s.npy' %name)
			print('Patches loaded')
		else:
			patches, patch_xyxy = image2patches(scales, img)
			np.save('patches_%s.npy' %name, patches)
			np.save('patch_position_%s.npy' %name, patch_xyxy)
			print('Patches saved')
		print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
		if os.path.exists('predicts_%s.npy' %name):
			predicts = np.load('predicts_%s.npy' %name)
			print('Predicts loaded')
		else:
			predicts = [self.sc_function(patch) for patch in tqdm(patches)]
			np.save('predicts_%s.npy' %name, predicts)
			print('Predicts saved')

		print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
		pos_predicts_xyxy = np.array([np.hstack((patch_xyxy[idx],np.array(score))) for idx, score in enumerate(predicts) if score > 0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		
		print('after nms:', xyxy_after_nms.shape[0])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #gree rectangular with line width 3


            
		return img

	def get_hard_negative_patches(self, img, scale_step = 10):
		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = np.array([self.sc_function(patch) for patch in tqdm(patches)])

		wrong_patches = patches[(predicts > 0)]

		return wrong_patches

	def visualize(self):
		self.visualizer.labels = self.labels
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		self.visualizer.draw_wc_accuracies()
		self.visualizer.draw_filters()
		self.visualizer.draw_err_strong()
