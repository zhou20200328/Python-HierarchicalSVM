from active_learning import *

def read_labels(fname):
	labels = []
	fd = open(fname)
	for line in fd:
		line = line.strip().split(',')
		id = line[0]
		v = line[1][1:(len(line[1])-1)].strip().split(' ')
		v = [int(vv) for vv in v]
		labels.append(set(v))
	fd.close()
	return labels

def read_label_lst(fname):
	labels = []
	fd = open(fname)
	for line in fd:
		line = line.strip()
		v = float(line)		
		labels.append(int(v))
	fd.close()
	return labels
	
def is_anc_in_set(l, root, parents, label_set):
	p = parents[l]
	while p != root:
		if p not in label_set:
			return False
		p = parents[p]
	return True
	
def make_predicted_labels(root, folder, fname_ext, test_labels, parents):
	for c in root.children:
		#read all examples' predicted labels for this node
		predicted_labels = read_label_lst(folder + '/' + str(c.labelIndex) + '.' + fname_ext)
		#update each examples' label set
		for i in range(len(test_labels)):
			if predicted_labels[i] == 1 and is_anc_in_set(c.labelIndex, 0, parents, test_labels[i]):
				test_labels[i].add(c.labelIndex)
		#go to deeper level
		make_predicted_labels(c, folder, fname_ext, test_labels, parents)
								
#output docs whose F1 score is less than 0.7						
def get_bad_hier_f1_docs(true_labels, predicted_labels, threshold):
	ex_size = len(true_labels)
	bad_ids = []
	good_ids= []	
	for i in range(ex_size):
		t_labels = true_labels[i]
		p_labels = predicted_labels[i]
		p_n = len(t_labels & p_labels)
		p = len(p_labels)
		t = len(t_labels)		
		#compute macro loss
		if p != 0:
			pre = float(p_n) / p
		else:
			pre = 0
		if t != 0:
			rec = float(p_n) / t
		else:
			rec = 0
		if pre != 0 and rec != 0:
			f1 = 2* pre * rec / (pre + rec)
		else:
			f1 = 0
		if f1 < threshold:
			bad_ids.append(i)
		else:
			good_ids.append(i)
	return bad_ids, good_ids
							
def compute_hier_f1(true_labels, predicted_labels):
	ex_size = len(true_labels)
	sum_p = 0
	sum_t = 0
	sum_p_t = 0
	sum_macro_pres = 0
	sum_macro_recs = 0
	sum_macro_f1s = 0
	for i in range(ex_size):
		t_labels = true_labels[i]
		p_labels = predicted_labels[i]
		p_n = len(t_labels & p_labels)
		p = len(p_labels)
		t = len(t_labels)
		#compute micro loss
		sum_p_t += p_n
		sum_p += p
		sum_t += t
		#compute macro loss
		if p != 0:
			pre = float(p_n) / p
		else:
			pre = 0
		if t != 0:
			rec = float(p_n) / t
		else:
			rec = 0
		if pre != 0 and rec != 0:
			f1 = 2* pre * rec / (pre + rec)
		else:
			f1 = 0
		sum_macro_pres += pre
		sum_macro_recs += rec
		sum_macro_f1s += f1
	#compute micro loss
	if sum_p != 0:
		micro_prec = float(sum_p_t) / sum_p
	else:
		micro_prec = 0
	if sum_t != 0:
		micro_rec = float(sum_p_t) / sum_t
	else:
		micro_rec = 0
	if micro_prec != 0 and micro_rec != 0:
		micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
	else:
		micro_f1 = 0
	#compute macro loss
	macro_prec = sum_macro_pres / ex_size
	macro_rec = sum_macro_recs / ex_size
	macro_f1 = sum_macro_f1s / ex_size
	
	return micro_prec, micro_rec, micro_f1, macro_prec, macro_rec, macro_f1

#compute hier f1 measure for top-k levels
def compute_hier_f1_in_top_k_levels(true_labels, predicted_labels, top_k_label_set):
	ex_size = len(true_labels)
	sum_p = 0
	sum_t = 0
	sum_p_t = 0
	sum_macro_pres = 0
	sum_macro_recs = 0
	sum_macro_f1s = 0
	for i in range(ex_size):
		t_labels = true_labels[i]
		p_labels = predicted_labels[i]
		#restrict the labels to the only top_k set
		t_labels = set([t for t in t_labels if t in top_k_label_set])
		p_labels = set([t for t in p_labels if t in top_k_label_set])
		p_n = len(t_labels & p_labels)
		p = len(p_labels)
		t = len(t_labels)
		#compute micro loss
		sum_p_t += p_n
		sum_p += p
		sum_t += t
		#compute macro loss
		if p != 0:
			pre = float(p_n) / p
		else:
			pre = 0
		if t != 0:
			rec = float(p_n) / t
		else:
			rec = 0
		if pre != 0 and rec != 0:
			f1 = 2* pre * rec / (pre + rec)
		else:
			f1 = 0
		sum_macro_pres += pre
		sum_macro_recs += rec
		sum_macro_f1s += f1
	#compute micro loss
	if sum_p != 0:
		micro_prec = float(sum_p_t) / sum_p
	else:
		micro_prec = 0
	if sum_t != 0:
		micro_rec = float(sum_p_t) / sum_t
	else:
		micro_rec = 0
	if micro_prec != 0 and micro_rec != 0:
		micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
	else:
		micro_f1 = 0
	#compute macro loss
	macro_prec = sum_macro_pres / ex_size
	macro_rec = sum_macro_recs / ex_size
	macro_f1 = sum_macro_f1s / ex_size
	
	return micro_prec, micro_rec, micro_f1, macro_prec, macro_rec, macro_f1
		
def init_test_labels(fname):
	#just make an empty slot list
	test_labels = []
	fd = open(fname)
	for line in fd:
		test_labels.append(set())
	fd.close()
	return test_labels
	
def get_doc_length(fname, ids):
	lends = []
	ids = set(ids)
	fd = open(fname)
	i = 0
	for line in fd:
		line = line.strip().split(' ')
		if i in ids:
			lends.append(len(line) - 1)
		i += 1 
	fd.close()
	return lends
	
def statistics_lends(lens):
	import math
	len_len = len(lens)
	tot_len = sum(lens)
	min_len = min(lens)
	max_len = max(lens)
	avg_len = float(tot_len) / len_len
	if len_len == 1:
		std_dev_len = 0
	else:
		std_dev_len = math.sqrt(sum([(v-avg_len)*(v-avg_len) for v in lens]) / (len_len - 1))
	return len_len, min_len, max_len, avg_len, tot_len
		
if __name__ == '__main__':
	folder = '/home/xiao/datasets/software/data'
	svm_folder = '/home/xiao/workspace/software/svm_models'
	svm_output_folder = '/home/xiao/workspace/software/svm_output'

	hier_fname = folder + '/' + 'sf_topics_nodag_id.txt'
	test_feature_fname = folder + '/' + 'sf_stemmed_testing_files_lx.svm'
	test_label_fname = folder + '/' + 'sf_stemmed_testing_tags_lx.svm'
	train_feature_fname = folder + '/' + 'sf_stemmed_training_files_lx.svm'
	train_label_fname = folder + '/' + 'sf_stemmed_training_tags_lx.svm'	

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
	#read predicted labels
	predicted_labels = init_test_labels(test_label_fname)
	#adjust the labels with tree constrain
	make_predicted_labels(root, svm_output_folder, 'test_labels', predicted_labels, parents)
	#output the loss
	print compute_hier_f1(true_labels, predicted_labels)
	
	#check the bad and good testing examples as splited by per-instance f1 threshold 
	threshold = 0.5
	bad_ids, good_ids = get_bad_hier_f1_docs(true_labels, predicted_labels, threshold)
	#get doc length for bad and good prediction
	bad_lengths = get_doc_length(test_feature_fname, bad_ids)
	good_lengths = get_doc_length(test_feature_fname, good_ids)
	print 'wrong', statistics_lends(bad_lengths)
	print 'correct', statistics_lends(good_lengths)
	
	#check loss for top-k level
	top_k_level_nodes = set()
	for d in range(levels):
		#get top-k level nodes
		top_k_level_nodes |= set(nodes_per_level[d])
		print 'top_k_levles', d,  compute_hier_f1_in_top_k_levels(true_labels, predicted_labels, top_k_level_nodes)
		
