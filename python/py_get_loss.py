def compute_hier_f1_skip_empty(true_labels, predicted_labels):
	ex_size = len(true_labels)
	sum_p = 0
	sum_t = 0
	sum_p_t = 0
	sum_macro_pres = 0
	sum_macro_recs = 0
	sum_macro_f1s = 0
	num_used = 0
	for i in range(ex_size):
		t_labels = true_labels[i]
		p_labels = predicted_labels[i]		
		p_n = len(t_labels & p_labels)
		p = len(p_labels)
		t = len(t_labels)
		if p == 0:
			continue
		num_used += 1
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
	macro_prec = sum_macro_pres / num_used
	macro_rec = sum_macro_recs / num_used
	macro_f1 = sum_macro_f1s / num_used
	
	return micro_prec, micro_rec, micro_f1, macro_prec, macro_rec, macro_f1, num_used, ex_size

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
		
def read_labels(fname):
	labels = []
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		id = line[0]
		v = line[1:]
		v = [int(vv) for vv in v]
		labels.append(set(v))
	fd.close()
	return labels
	
true_labels = read_labels('/home/xiao/workspace/enrich/freecode/test_labels.txt')
predicted_labels = read_labels('/home/xiao/workspace/enrich/sf_fr_prediction/predicted_labels.txt')
a = compute_hier_f1_skip_empty(true_labels, predicted_labels)
b = compute_hier_f1(true_labels, predicted_labels)
print a
print b

