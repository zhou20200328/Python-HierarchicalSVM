from active_learning import *
from py_compute_hierarchical_loss import *

def usage():
	print 'cmd --hier fname1 --telabel fname4 --predictionfolder fname5 --lossf fname'
	
if __name__ == '__main__':
	import getopt, sys
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'iot:h', ['help', 'hier=', 'telabel=', 'predictionfolder=', 'lossf='])
	except getopt.GetoptError, err:
		print 'err'
		usage()
		sys.exit(1)
	
	hier_fname = ''	
	test_label_fname = ''	
	model_output = ''
	prediction_output = ''
	loss_fname = ''
	
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
		elif opt in ('--lossf'):
			loss_fname = arg
		
	if hier_fname == '' or test_label_fname == '' or prediction_output == '' or loss_fname == '':
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
	make_predicted_labels_by_maximal_prob(root, prediction_output, 'test_probs', predicted_labels, parents)
	#output the loss
	hier_loss = compute_hier_f1(true_labels, predicted_labels)
	print hier_loss
	fw = open(loss_fname, 'w')
	fw.write(' '.join([str(l) for l in hier_loss]))
	fw.close()
	#write prdicted labels
	fd = open(prediction_output + '/predicted_labels_max_prob.txt', 'w')
	n = 0
	for p in predicted_labels:
		labels = list(p)
		labels.sort()
		labels = [str(l) for l in labels]
		fd.write(str(n) + ' ' + ' '.join(labels) + '\n')
		n += 1
	fd.close()
	
	"""
	#check loss for top-k level	
	top_k_level_nodes = set()
	for d in range(levels):
		#get top-k level nodes
		top_k_level_nodes |= set(nodes_per_level[d])
		print 'top_k_levles', d,  compute_hier_f1_in_top_k_levels(true_labels, predicted_labels, top_k_level_nodes)	
	"""
