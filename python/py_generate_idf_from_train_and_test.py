def get_maximal_words(fname):
	max_id = -1
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		line = [int(l) for l in line]
		v = max(line[1:])
		if v > max_id:
			max_id = v
	fd.close()
	return max_id
	
def build_idf_table(fname_lst):
	idf = {}
	tot_docs = 0
	for fname in fname_lst:
		fd = open(fname)
		for line in fd:
			line = line.strip().split(' ')
			tot_docs += 1
			#get distinct words
			all_words = set(line[1:])
			for word in all_words:
				if word in idf:
					idf[word] += 1
				else:
					idf[word] = 1				
		fd.close()
	return idf, tot_docs
	
def write_idf(idf, tot_docs, fname):
	import math
	words = idf.keys()
	words.sort()
	fd = open(fname, 'w')
	for wd in words:
		fd.write(wd + ' ' + str(idf[wd]) + ' ' + str(math.log(float(tot_docs)/idf[wd])) + '\n')
	fd.close()
	
def read_idf(fname):
	idf = {}
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		idf[line[0]] = float(line[2])
	fd.close()
	return idf

def count_all_docs(fname):
	n = 0
	fd = open(fname)
	for line in fd:
		n += 1
	fd.close()
	return n
	
def read_ids(fname):
	ids = []
	fd = open(fname)
	for line in fd:
		a = line.index(' ')
		ids.append(line[:a])
	fd.close()
	return ids
	
def check_ids(ids1, ids2):
	import sys
	if len(ids1) != len(ids2):
		print 'bad length'
		sys.exit(1)
	for i in range(len(ids1)):
		if ids1[i] != ids2[i]:
			print 'bda ids', ids1[i], ids2[i]
			sys.exit(1)

def usage():
	print 'cmd --traintxt fname1 --traintopic fname2 --testtxt fname3 --testtopic fname 4 --idf fname3'
	
if __name__ == '__main__':
	import getopt, sys
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'abcde:h', ['help', 'traintxt=', 'traintopic=', 'testtxt=', 'testtopic=', 'idf='])
	except getopt.GetoptError, err:
		print 'err'
		usage()
		sys.exit(1)
	
	in_train_txt = ''
	in_train_topic = ''
	in_test_txt = ''
	in_test_topic = ''
	out_idf = ''
	
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit(0)
		elif opt in ('--traintxt'):
			in_train_txt = arg
		elif opt in ('--traintopic'):
			in_train_topic = arg
		elif opt in ('--testtxt'):
			in_test_txt = arg
		elif opt in ('--testtopic'):
			in_test_topic = arg
		elif opt in ('--idf'):
			out_idf = arg
	if in_train_txt == '' or in_train_topic == '' or in_test_txt == '' or in_test_topic == '' or out_idf == '':
		usage()
		sys.exit(1)
			
	fname_lst = [in_train_txt, in_train_topic, in_test_txt, in_test_topic]
	
	#build idf table
	idf_table, tot_docs = build_idf_table(fname_lst)
	
	#write IDF table
	write_idf(idf_table, tot_docs, out_idf)	
