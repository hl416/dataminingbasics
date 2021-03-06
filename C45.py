############################################################################################
#决策树可能是最简单直观的一种分类方法，算法中涉及的理论也比较少。核心思路是：每次选择一个features
#，使得利用该feature来分割样本，能够得到最大的信息增益，也就是分割后得到的每组样本的熵最小（最稳定)
#熵越大，越混乱，越不稳定。
#决策树除了使用最大信息增益，也有方法是通过基尼等。
#C4.5与ID3在分拆节点上是一致的，都是利用最大信息增益；但C4.5在ID3的基础上，进行了很多优化，比如可以处理连续值（分区间），剪枝等。
#CART分拆节点利用的是基尼系数。
############################################################################################
import math

dict_tree = {}

def calc_entory(lst_data): #calcuate the entropy
	dict_label_count = {}
	for d in lst_data:
		label = d[-1]
		dict_label_count.setdefault(label,0)
		dict_label_count[label] += 1
	lst_prob = [dict_label_count[l]/float(len(lst_data)) for l in dict_label_count]
	print lst_prob
	return -sum([p * math.log(p,2) for p in lst_prob])

def split_data(lst_data,feature_idx,feature_value):
	return [d[:feature_idx]+d[feature_idx+1:] for d in lst_data if d[feature_idx] == feature_value]	#

def select_feature(lst_data):
	base_entroy = calc_entory(lst_data)
	max_gain = 0.0
	best_feature_idx = 0
	for i in range(0,len(lst_data[0])-1):
		cur_entropy = 0.0
		for value in set([d[i] for d in lst_data]):
			sub_data = split_data(lst_data,i,value)
			prob = len(sub_data)/float(len(lst_data))
			cur_entropy += calc_entory(sub_data) * prob #the entropy after spliting the entropies for each sub-group multiple their probability,respectively
		info_gain = base_entroy - cur_entropy
		if  info_gain > max_gain: #select the largest info gain
			max_gain = info_gain 
			best_feature_idx = i
	return best_feature_idx

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

def build_tree(lst_data):
	print lst_data
	lst_labels = [d[-1] for d in lst_data]
	if (lst_labels.count(lst_labels[0]) == len(lst_labels)):#only one catogory now
		print "only one category"
		return lst_labels[0]
	elif len(lst_data[0]) == 1: #all features have been used
		print "no features left"
		return major_label(lst_labels) #since we have no feature to use now, we return the label who dominates the current samples of the current sub-group 
	dict_tree_tmp = {}
	best_feature_idx = select_feature(lst_data)
	print "feature selected: %s" % str(set([d[best_feature_idx] for d in lst_data]))
	for value in set([d[best_feature_idx] for d in lst_data]):
		dict_tree_tmp[value] = build_tree(split_data(lst_data,best_feature_idx,value))
	return dict_tree_tmp


if __name__ == "__main__":
	lst_data = [d.strip('\n').split(',') for d in open("C4.5.data")]
	print calc_entory(lst_data)
	print build_tree(lst_data)
	
