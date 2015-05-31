#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

############################################################################################
#logistic regression可以理解为在linear regression的基础上加上了sigmoid函数的映射，把实数范围映射到了
#[0,1]这样一个空间。logistic_regression原则上还是解决线性可分的问题，如果点正好在分割平面上，则值为0；
#否则分别大于或者小于0，则对应两个不同类，也就是0和1.
#
#迭代的目的是减少cost function，也就是所谓的Error。但是这个Error是我们自己定义的，可以有不同的定义。
#1/m \sum(y-y')^2是一种比较经典的定义，在linear regression没有什么问题。但是在linear regresssion如果
#也采用同样的定义，则最后的cost function将不会是一个凸函数，也就会使得很容易出现局部最优，但是不是
#整体最后的情况。因此，在logistic regression里面，cost function的定义加上了log，使得最后的cost function
#为凸函数。cost(h(x),y) = -y log(h(x)) - (1-y) log(1-h(x)).
#当然，还有别的cost function也能得到凸函数，但是我们选取的这个函数还有计算上的优势。
#
#形式上，logistic_regression只能解决空间上线性可分的问题，因为它其实就是从线性回归加了一个转化函数。为了
#解决线性不可分的问题，比如二维空间里圆圈内部和外部分别对应两种类别，可以做特征变换，将地位的特征（对应
#圆圈就是两维）转化到高维（ [x1,x2]=>[x1,x2,x1^2,x2^2,x1x2])，这样就线性可分了。因此，我们在使用逻辑回归
#之前，是需要首先观察或者至少考虑在目前feature情况下，是否是线性可分的。如果不是，则需要做转换。
#
#如果feature本身就是线性可分的，不会有regularization的问题；但是如果feature本身不是，则由于特征转换，会带来
#overfiting的问题：你可以将10个feature转化为20个，也可以转换为100个，但是有可能20个就已经够了，你转换为
#100个虽然cost值更小，但是模型太复杂了，参数太多了，其实overfiting了，并不是对数据最好的描述。为了解决这个
#问题，尽可能放模型更简单的同时线性可分，我们在cost function加上了regularization这一项。有了这一项，一方面，
#如果我们通过特征转换又额外增加了一些特征，会导致这一项变大；另一方面，如果我们已经通过特征转换得到了100个
#特征，则最优化过程会使得每一个theta参数尽可能为0，最后的效果是减少了参数（及其对应的特征）的个数。
#
#gamma参数的大小决定了regularization这部分的权重，影响着模型的overfiting或者underfiting。在实际中可能需要
#进行尝试.
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
	m = len(X_train)   #number of samples
	n = len(X_train[0]) #number of features
	theta = zeros((n+1,1))
	X_train = concatenate((ones((m,1)),X_train),axis=1) #add a column of 1 to the left of the matrix, corresponding to theta_{0}
	Y_train = mat(Y_train).T #need to transfrom Y to the right dimensions
	J = float('inf');
	for i in range(0,ite_num):
		sig = sigmoid(dot(X_train , theta))
		err = sum(multiply(-Y_train,log(sig)) - multiply((1-Y_train), log(1-sig)))/m; # cost function: J = sum(-y .* log(sig) - (1-y) .* log(1-sig))/m;
		grad = transpose(dot(transpose(subtract(sig ,Y_train )),  X_train))/m #gradient
		theta -= multiply(alpha, grad) #update theta, this is exactly the same with linear regression
		if abs(err - J) < eps: #if no obvious difference between two iterations, then give up
			return
		J = err
		print "iter:%d error: %f" %(i,J)


if __name__ == "__main__":
	X,Y = make_classification(n_samples=400,n_informative=10,n_features=20,n_classes=2)
	print X,Y
	print sigmoid(X)
	X_train = X[:300]
	X_test = X[300:]
	Y_train = Y[:300]
	Y_test = Y[300:]
	logistic_regression(X_train,Y_train,3000,0.1,0.000001)
