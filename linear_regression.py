from sklearn.datasets.samples_generator import make_regression 
from scipy import stats

############################################################################################
#alpha: learning rate. too small: slow to converge; too large: cost function J may sometimes increase..not converge; So choose an alpha and make sure J decreases. In practice, choose 0.001,then 0.01,then 0.1 then 1. A figure with X-axis iter_num and Y-axis J can help you to decide.
#theta: 貌似没有固定套路，可以结合问题本身大概估计可能的最优值作为初始值。
#eps: this depends on what kind of precison you want and the time overhead you can bare
#normalizatino/scaling: help the algorithm to converge

#求解LR的另一种方法是正则方程Normal Equation.对于一个的函数，可以利用导数为0来求解最小值。与梯度下降这种迭代的算法相比，不需要选择各种参数，也不需要迭代的过程。当feature的数量n很小时，问题不大，但是当n很大时，则其需要大矩阵运算，复杂度为O(n^3)，非常慢。实际情况中，n<10000，可以选择normal equation.
############################################################################################

def batch_gradient_decent_one_variable(X,Y,eps,alpha,max_iter):
	m  = len(Y) # number of samples
	theta_0,theta_1 = [0,0] # initialization of theta
	J = 1.0/m * sum([(theta_0 + theta_1 * X[i] - Y[i]) ** 2 for i in range(0,m)])
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
