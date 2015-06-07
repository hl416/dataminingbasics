#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

############################################################################################
#neural networks可以看成是logistic regression的泛化。
#在neural networks中，依然使用梯度下降求解。但是由于梯度下降求的是误差的导数，对于输出层，
#当然没有什么问题，但是对于中间的隐藏层，则由于我们并不知道真实值，不好求误差。于是呢，我们
#采用后向传播法。从输出层开始，逐层往前计算误差。
############################################################################################

from sklearn.datasets.samples_generator import *
from scipy import stats
from sklearn import tree		
from numpy import *
lst_alpha = []
lst_models = []
lst_features_idx = []


def sigmoid(X):
	return 1./(1+exp(-X))

def logistic_regression(X_train,Y_train,ite_num,alpha,eps):

if __name__ == "__main__":
	X,Y = make_classification(n_samples=400,n_informative=10,n_features=20,n_classes=2)
	print X,Y
	print sigmoid(X)
	X_train = X[:300]
	X_test = X[300:]
	Y_train = Y[:300]
	Y_test = Y[300:]
	logistic_regression(X_train,Y_train,3000,0.1,0.000001)
