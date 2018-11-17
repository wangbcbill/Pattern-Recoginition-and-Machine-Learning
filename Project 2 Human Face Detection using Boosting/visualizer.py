import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Visualizer:
	def __init__(self, histogram_intervals, top_wc_intervals):
		self.histogram_intervals = histogram_intervals
		self.top_wc_intervals = top_wc_intervals
		self.weak_classifier_accuracies = {}
		self.strong_classifier_scores = {}
		self.Haar=None
		self.labels = None
	
	def draw_filters(self): 
		plt.figure(figsize=(10, 8))
		plt.clf()
		gs = gridspec.GridSpec(4, 5)
		gs.update(wspace=0.3, hspace=0.3)
    
		for i, haar in enumerate(self.Haar[0:20]):
			alpha=haar[0]
			wcs=haar[1]
			filt=np.repeat(0.5,16*16).reshape((16,16))
			filt[int(wcs.plus_rects[0][0]):int(wcs.plus_rects[0][2])+1,int(wcs.plus_rects[0][1]):int(wcs.plus_rects[0][3])+1]=1
			filt[int(wcs.minus_rects[0][0]):int(wcs.minus_rects[0][2])+1,int(wcs.minus_rects[0][1]):int(wcs.minus_rects[0][3])+1]=0
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			ax.set_title('weight = %.5f'%alpha)
			immin=(filt[:,:]).min()
			immax=(filt[:,:]).max()
			image=(filt-immin)/(immax-immin+1e-8)
			plt.imshow(image,cmap ='gray')
			plt.savefig('Top 20 Filters')
	
	def draw_histograms(self):
		for t in self.strong_classifier_scores:
			if t in self.histogram_intervals:
				scores = self.strong_classifier_scores[t]
				pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
				neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

				bins = np.linspace(np.min(scores), np.max(scores), 100)
				true_t=t+1   
                
				plt.figure()
				plt.hist(pos_scores, bins, alpha=0.5, label='Faces')
				plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces')
				plt.legend(loc='upper right')
				plt.title('Using %d Weak Classifiers' % true_t)
				plt.savefig('histogram_%d.png' % true_t)

	def draw_err_strong(self):
		plt.figure()
		err_strong=np.zeros(len(self.strong_classifier_scores))
		for t in self.strong_classifier_scores:        
			err_strong[t]=np.mean(np.sign(self.strong_classifier_scores[t])!=self.labels)
		plt.plot(np.arange(1,1+len(self.strong_classifier_scores)),err_strong)
		plt.ylabel('Error rate')
		plt.xlabel('steps')
		plt.title('Strong Classifier Training Error')
		plt.savefig('Strong Classifier Training Error')
        
	def draw_rocs(self):
		plt.figure()
		for t in self.strong_classifier_scores:
			if t in self.histogram_intervals:
				scores = self.strong_classifier_scores[t]
				fpr, tpr, _ = roc_curve(self.labels, scores)
				true_t=t+1   
				plt.plot(fpr, tpr, label = 'No. %d Weak Classifiers' % true_t)
		plt.legend(loc = 'lower right')
		plt.title('ROC Curve')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('ROC Curve')

	def draw_wc_accuracies(self):
		plt.figure()
		for t in self.weak_classifier_accuracies:
			accuracies = self.weak_classifier_accuracies[t]
			plt.plot(accuracies, label = 'After %d Selection' % t)
		plt.ylabel('Training Errors')
		plt.xlabel('Weak Classifiers')
		plt.title('Top 1000 Weak Classifier Training Errors')
		plt.legend(loc = 'lower right')
		plt.savefig('Weak Classifier Training Errors')

if __name__ == '__main__':
	main()
