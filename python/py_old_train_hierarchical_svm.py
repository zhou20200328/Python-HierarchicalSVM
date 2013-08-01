from liblinearutil_xiao import *
from active_learning import *
from check_dataset import *

def get_all_used_docs(fname, labelIndex, parents):
	docs = []
	n_index = 0
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]
		num_label = int(line[1])
		labels = line[2:]
		labels = [int(l) for l in labels]
		labels = set(labels)
		labels.add(-1)
		if labelIndex in labels:
			#positive
			docs.append((id, n_index, 1))
		else:#negative
			par = parents[labelIndex]
			if par == 0 or par in labels:
				docs.append((id, n_index, -1))
		n_index += 1
	fd.close()
	return docs

def read_problem_feature(docs, fname):
	id_used = set([d[0] for d in docs])	
	x = []	
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]
		if id in id_used:
			features = {}
			for l in line[1:]:
				wd, v = l.split(':')
				features[int(wd)] = float(v)
			x.append(features)
	fd.close()
	return x

def select_problem_TF_feature(indices, features):
	ret_features = []	
	for index in range(len(features)):
		if index in indices:			
			ret_features.append(features[index])
	return ret_features
	
def read_selected_problem_TF_feature(docs, fname):
	id_used = set([d[0] for d in docs])	
	x = []	
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]
		if id in id_used:			
			words = []
			occs = []
			for l in line[1:]:
				wd, v = l.split(':')
				words.append(int(wd))
				occs.append(int(v))
			#do normalization
			sm = sum(occs)
			occs = [float(o)/sm for o in occs]
			features = {}
			for i in range(len(words)):
				features[words[i]] = occs[i]
			#append examples			
			x.append(features)
	fd.close()
	return x

def get_max_feature(fname):
	max_f = -1
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]			
		for l in line[1:]:
			wd, v = l.split(':')
			if int(wd) > max_f:
				max_f = int(wd)		
	fd.close()
	return max_f

def read_problem_id(docs):
	ids = [int(d[0]) for d in docs]
	return ids

def read_problem_index(docs):
	indices = [int(d[1]) for d in docs]
	return indices
				
def read_problem_label(docs):
	y = [d[2] for d in docs]
	return y

def read_problem_feature(fname):	
	x = []
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]		
		features = {}		
		for l in line[1:]:
			wd, v = l.split(':')
			features[int(wd)] = float(v)
		x.append(features)
	fd.close()
	return x	

def read_problem_TF_feature(fname):	
	x = []
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]		
		words = []
		occs = []
		for l in line[1:]:
			wd, v = l.split(':')
			words.append(int(wd))
			occs.append(int(v))
		#do normalization
		sm = sum(occs)
		occs = [float(o)/sm for o in occs]
		features = {}
		for i in range(len(words)):
			features[words[i]] = occs[i]
		x.append(features)
	fd.close()
	return x
	
def write_list_to_file(data, fname):
	fd = open(fname, 'w')
	for d in data:
		v = '%.4f' % d
		fd.write(v + '\n')
	fd.close()

def get_silbing(c, parents):
	sib = []
	my_parent = parents[c]
	for nd in parents:
		if parents[nd] == my_parent:
			sib.append(nd)
	return sib

def read_used_probs(fname, used_indices):
	probs = []
	index = 0
	fd = open(fname)
	for line in fd:
		if index in used_indices:			
			p = float(line.strip())
			probs.append(p)
		index += 1
	fd.close()
	return probs
	
def read_used_probs_by_id(fname, used_id_set):
	probs = []	
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		did = int(line[0])
		prob = float(line[1])
		if did in used_id_set:			
			probs.append(prob)		
	fd.close()
	return probs

def read_probs(fname):
	probs = []	
	fd = open(fname)
	for line in fd:		
		p = float(line.strip())
		probs.append(p)
	fd.close()
	return probs
	
def prob_dict_to_lst(probs):
	all_cats = probs.keys()
	all_cats.sort()
	lst = []
	n_probs = len(probs[all_cats[0]])
	for i in range(n_probs):
		lst.append([probs[c][i] for c in all_cats])
	return lst

def compute_loss(probs, labels, threshold):
	tp = 0
	fp = 0
	fn = 0
	for i in range(len(probs)):
		if probs[i] >= threshold:
			p = 1
		else:
			p = -1
		l = labels[i]
		if p == 1 and l == 1:
			tp += 1
		elif p == 1 and l == -1:
			fp += 1
		elif p == -1 and l == 1:
			fn += 1
	if tp + fp != 0:
		pre = float(tp)/(tp + fp)
	else:
		pre = 0
	if tp + fn != 0:
		rec = float(tp)/(tp + fn)
	else:
		rec = 0
	if pre + rec == 0:
		f1 = 0
	else:
		f1 = 2* pre * rec / (pre + rec)
	return tp, fp, fn, pre, rec, f1
				
def usage():
	print 'cmd --hier fname1 --trfeature fname2 --trlabel fname3 --tefeature fname4 --telabel fname4 --modelfolder --predictionfolder --trainpredictionfolder'

def check_first_positive(x, y):
	if len(y) > 1:
		if y[0] == -1:
			#find the first positive
			for i in range(len(y)):
				if y[i] == 1:
					break
			if i < len(y):
				tmp = y[0]
				y[0] = y[i]
				y[i] = tmp
				tmp = x[0]
				x[0] = x[i]
				x[i] = tmp
				
def compute_loss_with_labels(pred_labels, labels):
	tp = 0
	fp = 0
	fn = 0
	for i in range(len(pred_labels)):
		if pred_labels[i] == 1:
			p = 1
		else:
			p = -1
		l = labels[i]
		if p == 1 and l == 1:
			tp += 1
		elif p == 1 and l == -1:
			fp += 1
		elif p == -1 and l == 1:
			fn += 1
	if tp + fp != 0:
		pre = float(tp)/(tp + fp)
	else:
		pre = 0
	if tp + fn != 0:
		rec = float(tp)/(tp + fn)
	else:
		rec = 0
	if pre + rec == 0:
		f1 = 0
	else:
		f1 = 2* pre * rec / (pre + rec)
	return tp, fp, fn, pre, rec, f1

def read_labels(fname):
	labels = []
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]
		num = int(line[1])
		v = line[2:]
		v = [int(vv) for vv in v]
		labels.append(set(v))
	fd.close()
	return labels


def get_binary_label_global(labels, label):
	ret_labels = []
	for ls in labels:		
		if label in ls:
			ret_labels.append(1)
		else:
			ret_labels.append(-1)
	return ret_labels
		
if __name__ == '__main__':
	import getopt, sys
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'iot:h', ['help', 'hier=', 'trfeature=', 'trlabel=', 'tefeature=', 'telabel=', 'modelfolder=', 'trainpredictionfolder=', 'predictionfolder='])
	except getopt.GetoptError, err:
		print 'err'
		usage()
		sys.exit(1)
	
	hier_fname = ''
	train_feature_fname = ''
	train_label_fname = ''
	test_feature_fname = ''
	test_label_fname = ''	
	model_output = ''
	prediction_output = ''
	trainpredictionfolder = ''
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit(0)
		elif opt in ('--hier'):
			hier_fname = arg
		elif opt in ('--trfeature'):
			train_feature_fname = arg
		elif opt in ('--trlabel'):
			train_label_fname = arg
		elif opt in ('--tefeature'):
			test_feature_fname = arg
		elif opt in ('--telabel'):
			test_label_fname = arg
		elif opt in ('--modelfolder'):
			model_output = arg
		elif opt in ('--predictionfolder'):
			prediction_output = arg
		elif opt in ('--trainpredictionfolder'):
			trainpredictionfolder = arg
		
	if hier_fname == '' or train_feature_fname == '' or train_label_fname == '' or test_feature_fname == '' or test_label_fname == '' or model_output == '' or prediction_output == '' or trainpredictionfolder == '':
		usage()
		sys.exit(1)
		
	#build hierarcy tree
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
	leaves = set(leaves)

	#get maximal feature
	max_f = get_max_feature(train_feature_fname)
	
	#read train features
	train_features = read_problem_TF_feature(train_feature_fname)
	
	#read test features
	test_features = read_problem_TF_feature(test_feature_fname)
	
	test_labels = read_labels(test_label_fname)
	
	threshold = 0.5
	
	#do training and testing from the top-level to the bottom level
	for cur_depth in range(levels):
	#for cur_depth in [1]:
		nodes = nodes_per_level[cur_depth]
		for l in nodes:#for nd in this level					
		#for l in [3]:
			#get <id, label> pairs 
			print 'train', l
			
			#localize data to each node
			docs = get_all_used_docs(train_label_fname, l, parents)
			
			#make training dataset		
			y = read_problem_label(docs)
			ids = read_problem_id(docs)
			indices = read_problem_index(docs)
				
			#x = read_problem_TF_feature(docs, train_feature_fname)
			#this is a new copy of the old instance list
			x = select_problem_TF_feature(set(indices), train_features)		
						
			#print statistics
			num_pos_y = y.count(1)
			num_neg_y = len(y) - num_pos_y
			print 'pos', num_pos_y, 'neg', num_neg_y, 'total', len(x)
			
			#check dataset, put the first element as positive +1
			check_first_positive(x, y)			
			#train SVM model					
			prob  = problem(y, x)
			param = parameter('-q')
			m = train(prob, param)			
			#save SVM model
			save_model(model_output + '/' + str(l) + '.svm', m)
			
			#make prediction on test							
			p_labs, p_acc, p_vals, p_probs = predict_label_score_prob([], test_features, m, '-q')
			#save prediction
			fd = open(prediction_output + '/' + str(l) + '_test_labels', 'w')
			for v in p_labs:
				fd.write(str(v) + '\n')
			fd.close()	
			fd = open(prediction_output + '/' + str(l) + '_test_probs', 'w')
			for v in p_probs:
				fd.write(str('%.4f' % v) + '\n')
			fd.close()					

			v_labels = []
			for p in p_probs:
				if p >= threshold:
					v_labels.append(1)
				else:
					v_labels.append(-1)
			true_bin_labels = get_binary_label_global(test_labels, l)
			print compute_loss_with_labels(v_labels, true_bin_labels)
