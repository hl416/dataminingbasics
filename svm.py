#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

############################################################################################
#SVM可以看做是从logistic regression的发展，是将logistic regresssion的cost function的等于1和0
#的条件变强，从而将两个区域到分割线的margin增加了。原始情况下，直接从logistic regression转换过来的SVM
#是没有kernel的，或者称为linear kernel。对于n(features数目）较大，m（样本数目）较小的情况，
#这种SVM已经能够比较好的解决问题，因为已经是线性可分的了。但是，对于n较小，m相对较大的情况，
#则由于不是线性可分，无法解决。于是，采用kernel增加维度/fetures，使得线性可分。例如，高斯kernel
#会将每一个样本点以基础，计算任意一个点到这个点的距离，作为一个feature，这样，m个点就会有m
#个features，从而达到线性可分。当然，还有其他的kernel可以使用，但是由于增加维度后计算的开销很大
#Kernel其实就是similarity function，但并不是所有similarity function都可以作为kernel。在计算相似度之前，一定要做feature scaling.
#对于同样的两个类别的点，logistic_regression分类的时候，目标是误差最小，而svm则是margin最大，因此，二者产生的分割线是不一样的。疑问：svm会不会对于outlier非常敏感
#SVM使用了很多trick来提高性能，这些trick并不是对所有的kernel适用。
#关于SVM的适用条件：1）如果n相对于m较大，则直接逻辑回归或者线性kernel；2）如果n较小，m适中，则高斯等Kernel；3）
#如果m很大，则一般先增加features，然后线性kernel或者逻辑回归，因为不然根本计算不出来，太慢了。线程kernel和
#logistic regression其实很像，二者适用条件类似。神经网络基本上述情况都可以用，但是计算远远慢于SVM和
#logsitic regression。
#SVM的参数选择：
#不像linear regression和logistic_regression都是没有限制的凸优化，直接梯度下降即可；
#SVM的求解是比较复杂的，SMO是一种比较流行的方法。
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
