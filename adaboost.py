#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

############################################################################################
#Adaboost的每一个weak learn在train的时候，是直接用training dataset训练得到的model来在同样的dataset上predict，计算误差/损失（线性回归其实也是这样）。
#alpha:每个weak learner的权重，重要的alpha更大。在最后得到的strong learner里面，可以考虑其不同weak learner的比重，也可以不考虑。在下面的代码中，两种情况均包含。
#iter_num: 如果迭代次数过多，可能overfiting，如果过少，不能学到足够的Pattern，准确性不好。因此，可以观察随着迭代次数增加，准确性的变化，然后在适当的时候停止迭代。当然，也有研究工作设计了regularization来解决，但是貌似不常用。
############################################################################################

from sklearn.datasets.samples_generator import *
from scipy import stats
from sklearn import tree		
from numpy import *
lst_alpha = []
lst_models = []
lst_features_idx = []

def major_label(lst_labels):
	dict_label_count = {}
	for l in lst_labels:
		dict_label_count.setdefault(l,0)
		dict_label_count[l] += 1
	max_cnt = 0
	max_label = ''
	for l in dict_label_count:
		if dict_label_count[l] > max_cnt:
			max_cnt = dict_label_count[l]
			max_label = l
	return max_label

def clear_models():
	lst_alpha = []
	lst_models = []
	lst_features_idx = []

def adaboost(X_train,Y_train,num_ite):
	clear_models()
	num_samples = len(Y_train)
	lst_weights = ones(num_samples)/num_samples

	for ite in range(0,num_ite):
		min_error = inf
		alpha = 0.0
		best_feature_idx = 0
		best_preditions = ones(num_samples)/num_samples
		best_model = None
		for feature_idx in range(0,len(X[0])): # search the best features to minimize the error, while this error gives more weights to the samples where the previous learners do not have good precision; each time we choose only one feature 
			X_train_tmp = X_train[:,feature_idx:feature_idx+1]
			M = tree.DecisionTreeClassifier(max_depth=1).fit(X_train_tmp,Y_train)
			Y_predict = M.predict(X_train_tmp) #we need to set max_depth, otherwise decision tree will have 100% precision on the training dataset as it will go down until the leaf
			error = [abs(l) for l in (Y_train - Y_predict)]
			e = sum(array(error)*lst_weights) #the errors for the previous classifier would affect the "defined" error of the current classifier,in this case, the current classifier will try to fix the error cases of the previous classifiers, i.e., minimizing e
			if (e < min_error):
				min_error = e
				best_feature_idx = feature_idx
				best_preditions = Y_predict
				best_model = M
				alpha = 0.5 * math.log((1-e)/e) # the coefficient for the current weak leaner, if the learner did well, then alpha is larger; otherwise small
		for i in range(0,num_samples): #update the weights for the training samples
			if best_preditions[i] == Y_train[i]:
				lst_weights[i] *= exp(-alpha)	#if the classification for one sample is correct, its weight decreases
			else:
				lst_weights[i] *= exp(alpha) #otherwise, its weight increases
		lst_weights = lst_weights/sum(lst_weights) #nomalization
		lst_alpha.append(alpha)
		lst_models.append(best_model) # store the decision tree model
		lst_features_idx.append(best_feature_idx) # which feature the decision tree model used

if __name__ == "__main__":
	X,Y = make_classification(n_samples=400,n_informative=50,n_features=60,n_classes=2)
	print X,Y
	X_train = X[:300]
	X_test = X[300:]
	Y_train = Y[:300]
	Y_test = Y[300:]

	for ite_num in [3,5,10,20,30,50,100,300,600,1000,2000,3000]: # we can compare the precision and designs the rule to stop: 1) the precision dreceases or 2) the precision descreases significantly (in case local optimal). It might be better to use cross-validation here.
		clear_models()
		print ite_num
		adaboost(X_train,Y_train,ite_num)
		lst_Y = []
		for j in range(0,len(lst_models)):
			lst_Y.append(lst_models[j].predict(X_test[:,lst_features_idx[j]:lst_features_idx[j]+1])) # 
		Y_predict_adaboost = []
		for i in range(0,len(Y_test)):
			Y_predict_adaboost.append(major_label([y[i] for y in lst_Y])) # just vote, do not consider the weight of each learner, i.e.,alpha
		#	Y_predict_adaboost.append(1 if sum([lst_Y[j][i]*lst_alpha[j] for j in range(0,len(lst_Y))])>=0.5 else 0) # vote with alpha considered
		print ite_num,float(sum([y == 0 for y in (Y_predict_adaboost - Y_test)]))/len(Y_test)
