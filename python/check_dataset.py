def checker(feature_file, label_file):
	f_id = []
	fd_f = open(feature_file)
	for line in fd_f:
		a = line.index(' ')
		f_id.append(int(line[:a]))
	fd_f.close()
	
	l_id = []
	fd_l = open(label_file)
	for line in fd_l:
		a = line.index(' ')
		l_id.append(int(line[:a]))
	fd_l.close()
	
	for i in range(len(f_id)):
		if f_id[i] != l_id[i]:
			print 'wrong ' + str(i)  + ' in ' + feature_file + ' and ' + label_file			
			
def checker2(feature_file, label_file):
	f_id = []
	n = 0
	flag_err = 0
	fd_f = open(feature_file)
	for line in fd_f:
		n += 1
		try:
			a = line.index(' ')
			f_id.append(int(line[:a]))
		except Exception:
			print 'error', line, ' at ', n			
			flag_err = 1
	fd_f.close()
	if flag_err == 1:
		return 
	l_id = []
	fd_l = open(label_file)
	for line in fd_l:
		a = line.strip().index(',')
		l_id.append(int(line[:a]))
	fd_l.close()
	
	for i in range(len(f_id)):
		if f_id[i] != l_id[i]:
			print 'wrong ' + str(i)  + ' in ' + feature_file + ' and ' + label_file			
			print f_id[i], l_id[i]
			return
			

def check_labels(file, label_size):
	fd = open(file)
	labels = [0 for i in range(label_size)]
	for line in fd:
		line = line.replace('\n','')
		line = line.split(' ')		
		for l in line[2:]:
			labels[int(l)] += 1
	fd.close()
	for i in range(len(labels)):
		if labels[i] == 0:
			print "Positive empty at " + str(i)
	return labels

def check_labels2(file, all_labels):
	fd = open(file)
	labels = {}
	label_docs = {}
	for l in all_labels:
		labels[l] = 0
		label_docs[l] = []
	n = 0
	for line in fd:
		n += 1
		line = line.strip().split(',')
		id = line[0]
		v = line[1][1:(len(line[1])-1)].strip()
		str_labels = v.split(' ')		
		try:	
			for l in str_labels:				
				labels[int(l)] += 1				
				label_docs[int(l)].append(id)
		except Exception:
			print l, line, n
			return
	fd.close()
	for l in all_labels:
		if labels[l] == 0:
			print "Positive empty at " + str(l)
	return labels, label_docs
	
def read_positive_count(file, label_size):
	fd = open(file)
	labels = [0 for i in range(label_size)]
	n = 0
	for line in fd:
		line = line.replace('\n','')
		line = line.split(' ')		
		n += 1
		for l in line[2:]:
			labels[int(l)] += 1
	fd.close()	
	return labels, n
	
def count(file1, label_size):
	tot_l=open(file1)
	l=[]
	for i in range(0,label_size):
		l.append(0)
	len1=0
	for line in tot_l:
		labels=line.strip().split(" ")					
		for label in labels[2:]:
			l[int(label)]+=1
	tot_l.close()
	return l

def check_ex_labels(hier_file, file):
	from active_learning import * 
	root = Node()
	root = root.read_tree(hier_file)
	max_depth = root.get_max_level()
	ns = []
	fd = open(file)
	ndoc = 0
	for line in fd:
		line = line.strip().split(' ')
		ns.append(int(line[1]))
		ndoc += 1
	fd.close()
	print ndoc, float(len([n for n in ns if n > max_depth]))/ndoc
	return ndoc, ns, float(len([n for n in ns if n > max_depth]))/ndoc

if __name__ == '__main__':	
	#check_labels('dmoz_small_labels.txt', 586)
	#n_rcv1 = check_ex_labels('rcv1-v2.hierarchy_train_96', 'lyrl2004_vectors_train_labels.dat')
	#n_ohsumed = check_ex_labels('ohsumed_hierarchy.txt', 'ohsumed_labels.txt')
	#n_dmoz = check_ex_labels('dmoz_hierarchy.txt', 'dmoz_small_labels.txt')
	#n_wipo = check_ex_labels('wipo_hierarchy.txt', 'wipo_labels.txt')

	#l1 = check_labels(workspace_folder + '/' + dataset_path + '/' + dataset + '_round_0_unlabeled_labels',72)	

	"""
	l1 = check_labels('1_20_labels',662)	
	print 'next'
	l2 = check_labels('2_80_labels',662)	
	print 'next'
	l2 = check_labels('D:\\dataset\\dmoz_full\\full\\dmoz\\dmoz_consist_labels_4.txt',662)	
	"""
	#checker('1_30_features', '1_30_labels')
	#checker('2_70_features', '2_70_labels')


	#checker('labeled_10_features','labeled_10_labels')
	#checker('unlabeled_70_features','unlabeled_70_labels')
	#checker('test_20_features','test_20_labels')
	"""
	checker('rcv1_small_unlabeled_features','rcv1_small_unlabeled_labels')
	checker('rcv1_small_labeled_features','rcv1_small_labeled_labels')
	checker('rcv1_small_test_features','rcv1_small_test_labels')
	"""
	"""
	ds = 'dmoz'
	for i in range(3):	
		#checker(ds + '_round_' + str(i) + '_labeled_features',ds + '_round_' + str(i) + '_labeled_labels')
		#checker(ds + '_round_' + str(i) + '_unlabeled_features',ds + '_round_' + str(i) + '_unlabeled_labels')
		#checker(ds + '_round_' + str(i) + '_test_features',ds + '_round_' + str(i) + '_test_labels')
		#check_labels(ds + '_round_' + str(i) + '_labeled_labels', 661)
		#check_labels(ds + '_round_' + str(i) + '_unlabeled_labels', 661)
		#check_labels(ds + '_round_' + str(i) + '_test_labels', 661)
	"""

	#l=count('dmoz_0.01_round_0_labeled_labels', 662)
