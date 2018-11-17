import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *
import os
import pickle

def main():
	#flag for debugging
	flag_subset = False
	boosting_type = 'Ada' #'Real' or 'Ada'
	training_epochs = 100 if not flag_subset else 20
	act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
	chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'

	#data configurations
	pos_data_dir = 'newface16'
	neg_data_dir = 'nonface16'
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
	data = integrate_images(normalize(data))

	#number of bins for boosting
	num_bins = 25

	#number of cpus for parallel computing
	num_cores = 8 if not flag_subset else 1 #always use 1 when debugging
	
	#create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

	#create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([9, 49, 99], [1, 10, 50, 100])
	
	#create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

	#calculate filter values for all training images
	start = time.clock()
	boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))

	boost.train(chosen_wc_cache_dir, chosen_wc_cache_dir)

	boost.visualize()
		 
#	original_img = cv2.imread('./Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
#	name="Face_1"
#	result_img = boost.face_detection(original_img,name)
#	cv2.imwrite('Result_img_%s.png' % name, result_img)
    
#	original_img = cv2.imread('./Testing_Images/Face_2.jpg', cv2.IMREAD_GRAYSCALE)
#	name="Face_2"
#	result_img = boost.face_detection(original_img,name)
#	cv2.imwrite('Result_img_%s.png' % name, result_img)
    
#	original_img = cv2.imread('./Testing_Images/Face_3.jpg', cv2.IMREAD_GRAYSCALE)
#	name="Face_3"
#	result_img = boost.face_detection(original_img,name)
#	cv2.imwrite('Result_img_%s.png' % name, result_img)

	for i in range(3):
		name = 'Face_%d' %(i+1)
		original_img = cv2.imread('./Testing_Images/Face_%d.jpg'%(i+1), cv2.IMREAD_GRAYSCALE)
		result_img = boost.face_detection(original_img,name)
		cv2.imwrite('Result_img_%s.png' % name, result_img)
        
	#negative mining    	
	if os.path.exists('new_data.npy'):
		new_data = np.load('new_data.npy')
	else:
		for i in range(3):
			name = 'Non_Face_%d' %(i+1)
			if os.path.exists('neg_patches_%s.npy'%name):
				neg_patches = np.load('neg_patches_%s.npy'%name)
			else:
				neg_img = cv2.imread('./Testing_Images/%s.jpg' %name, cv2.IMREAD_GRAYSCALE)
				neg_patches = boost.get_hard_negative_patches(neg_img)
				np.save('neg_patches_%s.npy'%name, neg_patches)
			if i==0:
				new_data = neg_patches
			else:           
				new_data = np.concatenate((new_data,neg_patches), axis=0)
			np.save('new_data.npy',new_data)
	n = len(new_data)
	print('Mined negative examples: %d' %n)

	new_data = np.concatenate((data, new_data),0)
	new_labels = np.concatenate((labels,-1*np.ones((n))),0)

	if os.path.exists('wc_activations_neg.npy'):
		wc_activations_neg = np.load('wc_activations_neg.npy')
	else:
		num_cores = 2
		act_cache_dir_neg = 'wc_activations_neg.npy' if not flag_subset else 'wc_activations_subset_neg.npy'
		#create boost classifier with a pool of weak classifier
		boost_neg = Boosting_Classifier(filters, new_data[57194:], new_labels[57194:], training_epochs, num_bins, drawer, num_cores, boosting_type)

		#calculate filter values for all training images
		start = time.clock()
		boost_neg.calculate_training_activations(act_cache_dir_neg, act_cache_dir_neg)
		end = time.clock()
		print('%f seconds for activation calculation' % (end - start))

	if os.path.exists('wc_activations_hard.npy'):
		wc_activations_hard = np.load('wc_activations_hard.npy')
	else:
		wc_activations=np.load('wc_activations.npy')
		wc_activations_neg=np.load('wc_activations_neg.npy')
		wc_activations_hard=np.concatenate((wc_activations,wc_activations_neg),axis=1)
		np.save('wc_activations_hard.npy',wc_activations_hard)
	
	act_cache_dir_hard = 'wc_activations_hard.npy' if not flag_subset else 'wc_activations_subset_hard.npy'
	chosen_wc_cache_dir_hard = 'chosen_wcs_hard.pkl' if not flag_subset else 'chosen_wcs_subset_hard.pkl'
	
	boost_hard = Boosting_Classifier(filters, new_data, new_labels, training_epochs, num_bins, drawer, num_cores, boosting_type)
	
	#calculate filter values for all training images
	start = time.clock()
	boost_hard.calculate_training_activations(act_cache_dir_hard, act_cache_dir_hard)
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))
       
	boost_hard.train(chosen_wc_cache_dir_hard, chosen_wc_cache_dir_hard)
#	original_img = cv2.imread('./Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
#	name="Face_1"
#	name_hard="Face_1_hard"

#	if os.path.exists('patches_%s.npy' %name_hard):
#		pass 
#	else:
#		patches = np.load('patches_%s.npy' %name)
#		patch_xyxy = np.load('patch_position_%s.npy' %name)
#		np.save('patches_%s.npy' %name_hard, patches)
#		np.save('patch_position_%s.npy' %name_hard, patch_xyxy) 

#	result_img_hard = boost_hard.face_detection(original_img,name_hard)
#	cv2.imwrite('Result_img_%s.png' % name_hard, result_img_hard)
    
#	original_img = cv2.imread('./Testing_Images/Face_2.jpg', cv2.IMREAD_GRAYSCALE)
#	name="Face_2"
#	name_hard="Face_2_hard"

#	if os.path.exists('patches_%s.npy' %name_hard):
#		pass 
#	else:
#		patches = np.load('patches_%s.npy' %name)
#		patch_xyxy = np.load('patch_position_%s.npy' %name)
#		np.save('patches_%s.npy' %name_hard, patches)
#		np.save('patch_position_%s.npy' %name_hard, patch_xyxy) 

#	result_img_hard = boost_hard.face_detection(original_img,name_hard)
#	cv2.imwrite('Result_img_%s.png' % name_hard, result_img_hard)
    	
#	original_img = cv2.imread('./Testing_Images/Face_3.jpg', cv2.IMREAD_GRAYSCALE)
#	name="Face_3"
#	name_hard="Face_3_hard"

#	if os.path.exists('patches_%s.npy' %name_hard):
#		pass 
#	else:
#		patches = np.load('patches_%s.npy' %name)
#		patch_xyxy = np.load('patch_position_%s.npy' %name)
#		np.save('patches_%s.npy' %name_hard, patches)
#		np.save('patch_position_%s.npy' %name_hard, patch_xyxy) 

#	result_img_hard = boost_hard.face_detection(original_img,name_hard)
#	cv2.imwrite('Result_img_%s.png' % name_hard, result_img_hard)

	for i in range(3):
		name = 'Face_%d' %(i+1)
		name_hard ='Face_%d_hard' %(i+1)
		original_img = cv2.imread('./Testing_Images/Face_%d.jpg'%(i+1), cv2.IMREAD_GRAYSCALE)
		if os.path.exists('patches_%s.npy' %name_hard):
			pass 
		else:
			patches = np.load('patches_%s.npy' %name)
			patch_xyxy = np.load('patch_position_%s.npy' %name)
			np.save('patches_%s.npy' %name_hard, patches)
			np.save('patch_position_%s.npy' %name_hard, patch_xyxy) 

		result_img_hard = boost_hard.face_detection(original_img,name_hard)
		cv2.imwrite('Result_img_%s.png' % name_hard, result_img_hard)
		
	#real boost
	drawer_real=Visualizer([9, 49, 99], None)
	drawer_real.labels=labels
	boost.ada2real(drawer_real)
	drawer_real.draw_histograms()
	drawer_real.draw_rocs()
if __name__ == '__main__':
	main()
