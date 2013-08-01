from active_learning import *

def read_labels(fname):
	labels = []
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]
		v = line[1:]#[1:(len(line[1])-1)].strip().split(' ')
		v = [int(vv) for vv in v]
		labels.append(set(v))
	fd.close()
	return labels

def read_labels_within_used(fname, all_labels):
	labels = []
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]
		num_label = int(line[1])
		v = [int(l) for l in line[2:] if int(l) in all_labels]		
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
	
def read_value_lst(fname):
	values = []
	fd = open(fname)
	for line in fd:
		line = line.strip()
		v = float(line)		
		values.append(v)
	fd.close()
	return values
	
def is_anc_in_set(l, root, parents, label_set):
	p = parents[l]
	while p != root:
		if p not in label_set:
			return False
		p = parents[p]
	return True
	
def make_predicted_labels_by_maximal_svm(root, folder, fname_ext, test_labels, parents):
	for c in root.children:
		#read all examples' predicted values for this node
		predicted_values = read_value_lst(folder + '/' + str(c.labelIndex) + '_' + fname_ext)
		#update each examples' label set
		for i in range(len(test_labels)):			
			if predicted_values[i] == 1 and is_anc_in_set(c.labelIndex, -1, parents, test_labels[i]):
				test_labels[i].add(c.labelIndex)
		#go to deeper level
		make_predicted_labels_by_maximal_svm(c, folder, fname_ext, test_labels, parents)

def make_predicted_labels_prob_threshold(root, folder, fname_ext, test_labels, parents, thresholds):
	for c in root.children:
		#read all examples' predicted values for this node
		predicted_values = read_value_lst(folder + '/' + str(c.labelIndex) + '.' + fname_ext)
		#update each examples' label set
		for i in range(len(test_labels)):			
			if predicted_values[i] >= thresholds[c.labelIndex] and is_anc_in_set(c.labelIndex, 0, parents, test_labels[i]):
				test_labels[i].add(c.labelIndex)
		#go to deeper level
		make_predicted_labels_prob_threshold(c, folder, fname_ext, test_labels, parents, thresholds)
		
def make_predicted_labels_by_maximal_prob(root, folder, fname_ext, test_labels, parents):
	if len(root.children) == 0:
		return
	#read the probs of all children
	probs_children = []
	for c in root.children:
		#read all examples' predicted values for this node
		predicted_values = read_value_lst(folder + '/' + str(c.labelIndex) + '.' + fname_ext)		
		probs_children.append(predicted_values)
	#find the maximal prob for an example
	#update each examples' label set
	for i in range(len(test_labels)):		
		#get the maximal prob of all children
		#for j in range(len(root.children)):
		#	print 'a',len(probs_children), 'b',len(probs_children[j])
		probs = [(probs_children[j][i], root.children[j].labelIndex) for j in range(len(root.children))]
		probs = sorted(probs, key=lambda s:s[0], reverse=True)		
		max_label = probs[0][1]
		#only care example who has this label			
		if is_anc_in_set(max_label, 0, parents, test_labels[i]):				
			test_labels[i].add(max_label)			
	#go to deeper level	
	for c in root.children:
		make_predicted_labels_by_maximal_prob(c, folder, fname_ext, test_labels, parents)

def make_predicted_labels_by_only_first_level_maximal_prob(root, folder, fname_ext_label, fname_ext_prob, test_labels, parents):
	if len(root.children) == 0:
		return
	#get labels of all children
	label_children = []
	for c in root.children:
		#read all examples' predicted values for this node
		predicted_values = read_label_lst(folder + '/' + str(c.labelIndex) + '.' + fname_ext_label)	
		label_children.append(predicted_values)
	#assign label to examples
	empty_ex = []
	for i in range(len(test_labels)):
		#get the label of this example
		get_label = 0
		for j in range(root.children):
			if label_children[j][i] == 1 and is_anc_in_set(root.children[j].labelIndex, 0, parents, test_labels[i]):
				get_label = 1
				test_labels[i].add(root.children[j].labelIndex)
		#if the label is empty book it
		if get_label == 0:
			empty_ex.append(i)	
	#check probs
	if root.labelIndex == 0:
		#read the probs of all children	
		probs_children = []
		for c in root.children:
			#read all examples' predicted values for this node
			predicted_values = read_value_lst(folder + '/' + str(c.labelIndex) + '.' + fname_ext_prob)		
			probs_children.append(predicted_values)
		#find the maximal prob for an example
		#update each examples' label set
		for i in empty_ex:		
			#get the maximal prob of all children			
			probs = [(probs_children[j][i], root.children[j].labelIndex) for j in range(len(root.children))]
			probs = sorted(probs, key=lambda s:s[0], reverse=True)		
			max_label = probs[0][1]
			#only care example who has this label			
			#if is_anc_in_set(max_label, 0, parents, test_labels[i]):
			test_labels[i].add(max_label)				
	#go to deeper level	
	for c in root.children:
		make_predicted_labels_by_only_first_level_maximal_prob(c, folder, fname_ext, test_labels, parents)
							
def make_predicted_labels(root, folder, fname_ext, test_labels, parents):
	for c in root.children:
		#read all examples' predicted labels for this node
		predicted_labels = read_label_lst(folder + '/' + str(c.labelIndex) + '_' + fname_ext)		
		#update each examples' label set
		for i in range(len(test_labels)):			
			if predicted_labels[i] == 1 and is_anc_in_set(c.labelIndex, -1, parents, test_labels[i]):
				test_labels[i].add(c.labelIndex)
		#go to deeper level
		make_predicted_labels(c, folder, fname_ext, test_labels, parents)
				
def get_length_hier_f1_relation(fname, true_labels, predicted_labels):
	ex_size = len(true_labels)
	leng_lst = []
	hier_f1_lst = []
	fd = open(fname)
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
		#get length of document
		line = fd.readline().strip().split(' ')
		doc_len = len(line[1:])
		leng_lst.append(doc_len)
		hier_f1_lst.append((pre, rec, f1))
	fd.close()
	return leng_lst, hier_f1_lst

def compute_pr_re_f1(tp, fp, fn):
	pr = 0
	re = 0
	f1 = 0
	if tp + fp == 0:
		pr = 0
	else:
		pr = float(tp)/(tp+fp)
	if tp + fn == 0:
		re = 0
	else:
		re = float(tp)/(tp+fn)
	if pr + re == 0:
		f1 = 0
	else:
		f1 = 2 * pr * re / (pr + re)
	return pr, re, f1
				
def compute_per_label_pr_re_f1(true_labels, predicted_labels, all_labels, parents, root):
	per_label_pr = {}
	per_label_re = {}
	per_label_f1 = {}
	for l in all_labels:
		tp = 0
		fp = 0
		fn = 0		
		if parents[l] == root:
			#use all example
			for i in range(len(true_labels)):
				tr_set = set(true_labels[i])
				pr_set = set(predicted_labels[i])
				if l in tr_set and l in pr_set:
					tp += 1
				elif l not in tr_set and l in pr_set:
					fp += 1
				elif l in tr_set and l not in pr_set:
					fn += 1
		else:
			#only example whose parent label is positive
			for i in range(len(true_labels)):
				tr_set = set(true_labels[i])
				pr_set = set(predicted_labels[i])
				if parents[l] in tr_set:
					if l in tr_set and l in pr_set:
						tp += 1
					elif l not in tr_set and l in pr_set:
						fp += 1
					elif l in tr_set and l not in pr_set:
						fn += 1			
		pr, re, f1 = compute_pr_re_f1(tp, fp, fn)
		per_label_pr[l] = pr
		per_label_re[l] = re
		per_label_f1[l] = f1
	return per_label_pr, per_label_re, per_label_f1
					
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
			
def usage():
	print 'cmd --hier fname1 --telabel fname4 --predictionfolder fname5 --lossfolder fname'
	
if __name__ == '__main__':
	import getopt, sys
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'iot:h', ['help', 'hier=', 'telabel=', 'predictionfolder=', 'lossfolder='])
	except getopt.GetoptError, err:
		print 'err'
		usage()
		sys.exit(1)
	
	hier_fname = ''	
	test_label_fname = ''	
	model_output = ''
	prediction_output = ''
	lossfolder_fname = ''
	
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit(0)
		elif opt in ('--hier'):
			hier_fname = arg				
		elif opt in ('--telabel'):
			test_label_fname = arg
		elif opt in ('--predictionfolder'):
			prediction_output = arg
		elif opt in ('--lossfolder'):
			lossfolder_fname = arg
		
	if hier_fname == '' or test_label_fname == '' or prediction_output == '' or lossfolder_fname == '':
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
	actual_labels = set([l for l in all_labels if l != -1])
	true_labels = read_labels_within_used(test_label_fname, actual_labels)
	
	#init predicted labels
	predicted_labels = init_test_labels(test_label_fname)
	#adjust the labels with tree constrain
	make_predicted_labels(root, prediction_output, 'test_labels', predicted_labels, parents)
	#output the loss
	hier_loss = compute_hier_f1(true_labels, predicted_labels)
	print hier_loss
	fw = open(lossfolder_fname + '/hier_f1', 'w')
	fw.write(' '.join([str(l) for l in hier_loss]))
	fw.close()
	#write prdicted labels
	fd = open(lossfolder_fname + '/predict_labels_fold_0_0.5.txt', 'w')
	n = 0
	for p in predicted_labels:
		labels = list(p)
		labels.sort()
		labels = [str(l) for l in labels]
		fd.write(str(n) + ' ' + str(len(labels)) + ' ' + ' '.join(labels) + '\n')
		n += 1
	fd.close()
	
	#check loss for top-k level	
	top_k_level_nodes = set()
	for d in range(levels):
		#get top-k level nodes
		top_k_level_nodes |= set(nodes_per_level[d])
		print 'top_k_levles', d,  compute_hier_f1_in_top_k_levels(true_labels, predicted_labels, top_k_level_nodes)	

	#compute per lable pr, re and f1
	per_label_pr, per_label_re, per_label_f1 = compute_per_label_pr_re_f1(true_labels, predicted_labels, actual_labels, parents, root.labelIndex)
	fw = open(lossfolder_fname + '/per_label_f1', 'w')
	for l in actual_labels:
		fw.write(str(l) + ' ' + str(per_label_pr[l]) + ' ' + str(per_label_re[l]) + ' ' + str(per_label_f1[l]) + '\n')
	fw.close()
