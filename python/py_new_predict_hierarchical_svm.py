from liblinearutil_xiao import *
from active_learning import *
from check_dataset import *

def get_all_used_docs(fname, labelIndex, parents):
	docs = []
	fd = open(fname)
	for line in fd:
		line = line.strip().split(',')
		id = line[0]
		labels = line[1][1:(len(line[1]) - 1)].strip().split(' ')		
		labels = [int(l) for l in labels]
		labels = set(labels)
		if labelIndex in labels:
			#positive
			docs.append((id, 1))
		else:#negative
			par = parents[labelIndex]
			if par == 0 or par in labels:
				docs.append((id, - 1))
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
	
def read_problem_label(docs):
	y = [d[1] for d in docs]
	return y

def read_test_problem_feature(fname):	
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

def write_list_to_file(data, fname):
	fd = open(fname, 'w')
	for d in data:
		v = '%.4f' % d
		fd.write(v + '\n')
	fd.close()
			
def usage():
	print 'cmd --hier fname1 --tefeature fname4 --modelfolder --predictionfolder'
	
if __name__ == '__main__':
	import getopt, sys
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'iot:h', ['help', 'hier=', 'tefeature=', 'modelfolder=', 'predictionfolder='])
	except getopt.GetoptError, err:
		print 'err'
		usage()
		sys.exit(1)
	
	hier_fname = ''		
	test_feature_fname = ''	
	model_output = ''
	prediction_output = ''
	
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit(0)
		elif opt in ('--hier'):
			hier_fname = arg		
		elif opt in ('--tefeature'):
			test_feature_fname = arg		
		elif opt in ('--modelfolder'):
			model_output = arg
		elif opt in ('--predictionfolder'):
			prediction_output = arg
		
	if hier_fname == '' or test_feature_fname == '' or model_output == '' or prediction_output == '':
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

	#read test features
	test_features = read_test_problem_feature(test_feature_fname)
	
	for l in all_labels:	
		if l != 0:		
			#get <id, label> pairs 			
			#load SVM model
			m = load_model(model_output + '/' + str(l) + '.svm')
			#make prediction
			print 'test', l
			p_labs, p_acc, p_vals, p_probs = predict_label_score_prob([], test_features, m, '-q')
			#save prediction
			fd = open(prediction_output + '/' + str(l) + '.test_labels', 'w')
			for v in p_labs:
				fd.write(str(v) + '\n')
			fd.close()	
			fd = open(prediction_output + '/' + str(l) + '.test_probs', 'w')
			for v in p_probs:
				fd.write(str('%.4f' % v) + '\n')
			fd.close()
