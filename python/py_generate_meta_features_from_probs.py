from py_generate_idf_from_train_and_test import *
	
def generate_tf_svm(in_txt, out_svm):
	import math
	fd_t = open(in_txt)	
	fw = open(out_svm, 'w')	
	for line in fd_t:
		line_txt = line.strip().split(' ')		
		#get doc id
		doc_id = line_txt[0]
		#get all words
		basic_text = [l for l in line_txt[1:]]		
		#count words occurence
		words = {}		
		tot_words = len(basic_text)
		for txt in basic_text:
			if txt not in words:
				words[txt] = 1
			else:
				words[txt] += 1		
		#compute TF*IDF as tf(word, doc) * log(|D|/idf(word))
		word_names = []
		word_ids = []
		occs = []		
		for wd in words:
			word_names.append(wd)
			occs.append(words[wd])			
			word_ids.append(int(wd))
		all_words = []
		for i in range(len(word_names)):
			word = word_names[i]
			occ = occs[i]
			v = float(occ)			
			all_words.append((word_ids[i], v))
		all_words = sorted(all_words, key=lambda s:s[0])
		str_out_lst = [str(w[0]) + ':' + ('%.4f' % w[1]) for w in all_words]
		fw.write(doc_id + ' ' + ' '.join(str_out_lst) + '\n')
	fd_t.close()	
	fw.close()
	
def usage():
	print 'cmd --folder a --ext b --outsvm fname4'

def get_num_docs(fname):
	n = 0
	fd = open(fname)
	for line in fd:
		n += 1
	fd.close()
	return n

def read_probs(byte_per_line, n_start_pos, num_docs_to_read, folder, all_labels, fext):
	ex_probs = [[0 for j in range(len(all_labels))] for n in range(num_docs_to_read)]
	label_probs = [[] for j in range(len(all_labels))]
	#byte_per_line = 7
	#n_start_pos = 10
	#read probs based on label 
	for i in range(len(all_labels)):
		label = all_labels[i]
		fname = folder + '/' + str(label) + '.' + fext
		fd = open(fname)
		fd.seek(byte_per_line * n_start_pos)
		for k in range(num_docs_to_read):
			line = fd.readline()
			line = float(line.strip())		
			label_probs[i].append(line)
		fd.close()
	#convert label based storage to example based storage
	for ex in range(num_docs_to_read):
		for i in range(len(all_labels)):
			ex_probs[ex][i] = label_probs[i][ex]
	return ex_probs

def write_data_to_file(start_id, ex_probs, fa):
	n_docs = len(ex_probs)
	for e in ex_probs:
		data = [str(i+1)+':'+('%.4f' % e[i]) for i in range(len(e))]
		fa.write(str(start_id) + ' ' + ' '.join(data) + '\n')	
		start_id += 1
	return start_id	
		
if __name__ == '__main__':
	import getopt, sys, os
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'iot:h', ['help', 'folder=', 'ext=', 'outsvm='])
	except getopt.GetoptError, err:
		print 'err'
		usage()
		sys.exit(1)
	
	folder = ''	
	ext = ''
	outsvm = ''
	
	num_trunk_size = 1000
	byte_per_line = 7
	n_start_pos = 10
	
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit(0)
		elif opt in ('--folder'):
			folder = arg		
		elif opt in ('--ext'):
			ext = arg
		elif opt in ('--outsvm'):
			outsvm = arg
	if folder == '' or ext == '' or outsvm == '':
		usage()
		sys.exit(1)
	
	#get tottal number of files
	files = os.listdir(folder)
	files = [f for f in files if f.endswith(ext)]
	labels = []
	for f in files:
		b = f.index('.')
		labels.append(int(f[:b]))
	labels.sort()	
	
	#get document length
	num_docs = get_num_docs(folder + '/' + str(labels[0]) + '.' + ext)
	#get the total number of trunks and each trunk size
	num_trunks = num_docs / num_trunk_size
	last_trunk_size = num_docs % num_trunk_size
	if last_trunk_size != 0:
		num_trunks += 1
	else:
		last_trunk_size = num_trunk_size
	
	#write features
	fw = open(outsvm, 'w')
	doc_id = 0
	for trunk_id in range(num_trunks):	
		if trunk_id != num_trunks - 1:
			ex_probs = read_probs(byte_per_line, trunk_id * num_trunk_size, num_trunk_size, folder, labels, ext)	
		else:
			ex_probs = read_probs(byte_per_line, trunk_id * num_trunk_size, last_trunk_size, folder, labels, ext)	
		doc_id = write_data_to_file(doc_id, ex_probs, fw)
	fw.close()
