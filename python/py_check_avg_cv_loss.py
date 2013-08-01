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
	return len_len, min_len, max_len, avg_len, std_dev_len, tot_len
	
folder = 5
loss_folder = '/home/xiao/workspace/enrich/5fold_loss'
pr = []
re = []
f1 = []
for f in range(folder):
	fd = open(loss_folder + '/loss_fold_'+str(f)+'/hier_f1.txt')
	ct = fd.read()
	fd.close()	
	ct = ct.strip().split(' ')
	ct = [float(l) for l in ct]
	pr.append(ct[0])
	re.append(ct[1])
	f1.append(ct[2])
st_pr = statistics_lends(pr)
st_re = statistics_lends(re)
st_f1 = statistics_lends(f1)
print st_f1[3], st_f1[4]
