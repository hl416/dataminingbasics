#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

############################################################################################
#alpha: learning rate. too small: slow to converge; too large: cost function J may sometimes increase..not converge; So choose an alpha and make sure J decreases. In practice, choose 0.001,then 0.01,then 0.1 then 1. A figure with X-axis iter_num and Y-axis J can help you to decide.
#theta: 貌似没有固定套路，可以结合问题本身大概估计可能的最优值作为初始值。
#eps: this depends on what kind of precison you want and the time overhead you can bare
#normalizatino/scaling: 一个原因是让每个feature的作用平均化，不至于因为某个feature的范围大，最后使得error受这个feature影响过大；另一个原因是这样处理后，得到的theta的图形相对规则，收敛会比较快，不会出现比较畸形(skewed)的效果。比如，对于y=16x^2这样一个图，我们步长alpha希望选得短一些，因为梯度变化已经很大了，以防止跳过了最低值，而对于y=1/16 x^2这样一个图，由于梯度变化率比较小，则希望步长alpha大一些，以快速收敛。如果不normalization，则alpha其实对于各个维度是存在选择矛盾的，有的希望大，有的希望小，为了保证收敛，只能选择最小的那个值，最后导致某些维度收敛非常慢。如果每个维度都是y=x^2，那各个维度的希望就其实是一致了的。当然，如果不做normalization，可以针对每个维度单独设置自己的alpha，但这样不是麻烦吗？本来alpha这个参数就要尝试，不一定好设置，将alpha这个本来是标量的参数变为矢量就更麻烦了。

#在梯度下降中，虽然都是沿着梯度方向变化，由于乘以的系数alpha是定值，在梯度小的地方，我们变化慢，大的地方，变化快。这样是合理的：既想速度到达坡低，同时也不想走快了，错过了坡底。理论上，我们也可以只使用梯度的方向，而每次移动的步长不变。但是这样显然效果不会很好，比如，会在梯度很大的时候，移动太慢了；同时快到坡低，梯度很小的时候，不容易收敛，很容易错过最优值。

#求解LR的另一种方法是正则方程Normal Equation.对于一个的函数，可以利用导数为0来求解最小值。与梯度下降这种迭代的算法相比，不需要选择各种参数，也不需要迭代的过程。当feature的数量n很小时，问题不大，但是当n很大时，则其需要大矩阵运算，复杂度为O(n^3)，非常慢。实际情况中，n<10000，可以选择normal equation.
############################################################################################

from sklearn.datasets.samples_generator import make_regression 
from scipy import stats

def batch_gradient_decent_one_variable(X,Y,eps,alpha,max_iter):
	m  = len(Y) # number of samples
	theta_0,theta_1 = [0,0] # initialization of theta
	J = 1.0/m * sum([(theta_0 + theta_1 * X[i] - Y[i]) ** 2 for i in range(0,m)]) #cost function
	for ite in range(0,max_iter):
		grad_0 = 1.0/m * sum([(theta_0 + theta_1 * X[i] - Y[i]) for i in range(0,m)])
		grad_1 = 1.0/m * sum([(theta_0 + theta_1 * X[i] - Y[i]) * X[i] for i in range(0,m)])
		theta_0 -= alpha * grad_0
		theta_1 -= alpha * grad_1
		error = 1.0/m * sum([(theta_0 + theta_1 * X[i] - Y[i]) ** 2 for i in range(0,m)])
		print "iter:%d, error:%f" % (ite,error)
		if abs(J-error) <= eps: # if we do not see obvious difference in two sequential gradient decent opeartions
			return (theta_0,theta_1)
		J = error
	return


if __name__ == "__main__":
	X,Y=make_regression(n_samples=100,n_features=1,n_informative=1)
	print X,Y	
	print batch_gradient_decent_one_variable(X,Y,0.001,0.01,1000)
	# compare our results with stats 
	results_stats = stats.linregress(X[:,0], Y) 
	print 'theta_0: %s , theta_1 : %s' %(results_stats[1], results_stats[0]) 
