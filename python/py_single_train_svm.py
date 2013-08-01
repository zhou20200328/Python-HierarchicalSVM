from liblinearutil_xiao import *
from active_learning import *
from check_dataset import *
import math

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
	
def read_problem_label(fname):
	ls = []
	fd = open(fname)
	for line in fd:
		ls.append(int(line.strip()))
	fd.close()
	return ls

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

def read_feature_map(fname):
	f2map = {}
	fd = open(fname)
	for line in fd:
		line = line.strip().split(':')
		old_f = int(line[0])
		new_f = int(line[1])
		f2map[old_f] = new_f
	fd.close()
	return f2map

def remap_feature(x, f2map):
	new_x = []
	for xx in x:
		new_xx = {}
		for w in xx:
			if w in f2map:
				new_xx[f2map[w]] = xx[w]
		new_x.append(new_xx)
	return new_x
					
feature_fname = '/home/mpi/topic_model_svm/tmp_feature_0'
label_fname = '/home/mpi/topic_model_svm/tmp_label_0'
test_feature_fname = '/home/mpi/topic_model_svm/20_news/20_news_0_fold_test_text.svm'
test_label_fname = '/home/mpi/topic_model_svm/20_news/20_news_0_fold_test_label'

feature_map_fname = '/home/mpi/shareddir/topic_model/20_news/fold_0/models/0_feature_indices_fold_0'
model_fname = '/home/mpi/shareddir/topic_model/20_news/fold_0/models/0_model_fold_0'
model_add_fname = '/home/mpi/shareddir/topic_model/20_news/fold_0/models/0_model_fold_0_addtional.txt'

#read training data
train_x = read_problem_feature(feature_fname)
train_y = read_problem_label(label_fname)

#read feature map
feature_map = read_feature_map(feature_map_fname)

#check dataset, put the first element as positive +1
print 'num ex', len(train_x)
check_first_positive(train_x, train_y)			
#train SVM model					
prob  = problem(train_y, train_x)
param = parameter('-q')
m = train(prob, param)			

test_features = read_problem_TF_feature(test_feature_fname)
test_labels = read_labels(test_label_fname)

true_bin_labels = get_binary_label_global(test_labels, 0)

#map test features
mapped_test_features = remap_feature(test_features, feature_map)
p_labs, p_acc, p_vals, p_probs = predict_label_score_prob([], mapped_test_features, m, '-q')
new_labs = []
for p in p_probs:	
	if p > 0.5:
		new_labs.append(1)
	else:
		new_labs.append(-1)	
print compute_loss_with_labels(p_labs, true_bin_labels)
print compute_loss_with_labels(new_labs, true_bin_labels)
#print compute_loss_with_labels(p_labs, true_bin_labels)

"""
#read model
tmp_m = load_model(model_fname)
fd = open(model_add_fname)
line = fd.readline()
line = fd.readline()
fd.close()
line = line.strip().split(' ')
tmp_m.probA = float(line[1])
tmp_m.probB = float(line[2])

#p_labs, p_acc, p_vals, p_probs = predict_label_score_prob([], mapped_test_features, tmp_m, '-q')
p_labs, p_acc, p_vals, p_probs = predict_label_score_prob([], mapped_test_features, m, '-q')
new_probs = []
new_labs = []
for v in p_vals:
	p = 1.0/(1+math.exp(tmp_m.probA * v+tmp_m.probB))
	if p > 0.5:
		new_labs.append(1)
	else:
		new_labs.append(-1)	
print compute_loss_with_labels(new_labs, true_bin_labels)

"""
