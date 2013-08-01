from py_compute_hierarchical_loss import *

def usage():
	print 'cmd --hier fname1 --tefeature fname2 --telabel fname4 --predictionfolder fname5 --lossf fname'

def find_bin(v, bins):
	my_bin = 0
	for i in range(len(bins)):
		if v <= bins[i]:
			return i
		my_bin += 1
	return i + 1

def get_stat(lst):
	import math
	min_v = min(lst)
	max_v = max(lst)
	sum_v = sum(lst)
	avg_v = float(sum_v) / len(lst)
	if len(lst) == 1:
		std_dv = 0
	else:
		std_dv = math.sqrt(sum([(l-avg_v)*(l-avg_v) for l in lst]))/(len(lst)-1)
	return min_v, max_v, avg_v, std_dv
	
if __name__ == '__main__':
	import getopt, sys
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'iot:h', ['help', 'hier=', 'telabel=', 'predictionfolder=', 'lossf=', 'tefeature='])
	except getopt.GetoptError, err:
		print 'err'
		usage()
		sys.exit(1)
	
	hier_fname = ''	
	test_label_fname = ''	
	test_feature_fname = ''
	model_output = ''
	prediction_output = ''
	loss_fname = ''
	
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit(0)
		elif opt in ('--hier'):
			hier_fname = arg				
		elif opt in ('--tefeature'):
			test_feature_fname = arg
		elif opt in ('--telabel'):
			test_label_fname = arg
		elif opt in ('--predictionfolder'):
			prediction_output = arg
		elif opt in ('--lossf'):
			loss_fname = arg
		
	if hier_fname == '' or test_label_fname == '' or prediction_output == '' or loss_fname == '' or test_feature_fname == '':
		usage()
		sys.exit(1)
	
	#read tree			
	root, all_nodes = Node().read_parent_child_pair_tree(hier_fname)
	all_labels = all_nodes.keys()
	tree_size = root.get_tree_size() - 1
	levels = root.get_max_level()
	nodes_per_level = [[] for i in range(levels)]
	parents = {}
	nd_leaves = []
	root.get_nodes_per_level(0, nodes_per_level)
	root.get_leaves(nd_leaves)
	root.get_parents(parents)
	leaves = [l.labelIndex for l in nd_leaves]
	print tree_size, levels
	for i in range(levels):
		print i, len(nodes_per_level[i])
	print len(leaves)

	#read true lables
	true_labels = read_labels(test_label_fname)
	#init predicted labels
	predicted_labels = init_test_labels(test_label_fname)
	#adjust the labels with tree constrain
	make_predicted_labels(root, prediction_output, 'test_labels', predicted_labels, parents)
	#get the length and hier f1 loss relation
	leng_lst, hier_loss_lst = get_length_hier_f1_relation(test_feature_fname, true_labels, predicted_labels)
	hier_loss = compute_hier_f1(true_labels, predicted_labels)	
	print 'hierarchical loss', hier_loss
	
	#split the length into two parts, compute statistics
	min_len = min(leng_lst)
	max_len = max(leng_lst)
	leng_bins = range(min_len, max_len, 10)
	pres = [[] for i in range(min_len, max_len, 10)] + [[]]
	recs = [[] for i in range(min_len, max_len, 10)] + [[]]
	f1s = [[] for i in range(min_len, max_len, 10)] + [[]]
	for th in range(min_len, max_len, 10):
		#get the length less and equal than the threshold
		#and the length larger than the threshold
		sum_pr_leq = 0
		sum_re_leq = 0
		sum_f1_leq = 0
		n_leq = 0
		sum_pr_gq = 0
		sum_re_gq = 0
		sum_f1_gq = 0
		n_gq = 0
		for i in range(len(leng_lst)):
			if leng_lst[i] <= th:
				sum_pr_leq += hier_loss_lst[i][0]
				sum_re_leq += hier_loss_lst[i][1]
				sum_f1_leq += hier_loss_lst[i][2]
				n_leq += 1
			else:
				sum_pr_gq += hier_loss_lst[i][0]
				sum_re_gq += hier_loss_lst[i][1]
				sum_f1_gq += hier_loss_lst[i][2]
				n_gq += 1
			#find the bin
			my_bin = find_bin(leng_lst[i], leng_bins)
			pres[my_bin].append(hier_loss_lst[i][0])
			recs[my_bin].append(hier_loss_lst[i][1])
			f1s[my_bin].append(hier_loss_lst[i][2])
		avg_pr_leq = float(sum_pr_leq) / n_leq
		avg_re_leq = float(sum_re_leq) / n_leq
		avg_f1_leq = float(sum_f1_leq) / n_leq
 		avg_pr_gq = float(sum_pr_gq) / n_gq
		avg_re_gq = float(sum_re_gq) / n_gq
		avg_f1_gq = float(sum_f1_gq) / n_gq
 		print th, 'short', avg_pr_leq, avg_re_leq, avg_f1_leq, 'long', avg_pr_gq, avg_re_gq, avg_f1_gq
	for i in range(len(leng_bins)+1):
		data_pre = get_stat(pres[i])
		data_rec = get_stat(recs[i])
		data_f1 = get_stat(f1s[i])
		if i < len(leng_bins):			
			print i, leng_bins[i], len(pres[i]), data_pre[2], data_rec[2], data_f1[2]
		else:
			print i, 'other', len(pres[i]), data_pre[2], data_rec[2], data_f1[2]
