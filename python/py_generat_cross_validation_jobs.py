import os
data_folder = '/home/xiao/workspace/enrich/5folddataset'
data_prefix = 'train_sourceforge'
model_folder = '/home/xiao/workspace/enrich/5fold_models'
prediction_folder = '/home/xiao/workspace/enrich/5fold_prediction'
loss_folder = '/home/xiao/workspace/enrich/5fold_loss'
#hier_file = '/home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt'
hier_file = '/home/xiao/datasets/OSS_data/sf_pairwise_first_level_hierarchy'
folder = 5
fw = open('sh_all_job.sh', 'w')
for f in range(folder):	
	try:
		os.mkdir(model_folder + '/model_fold_' + str(f))
	except Exception:
		print 'exist'
	try:
		os.mkdir(prediction_folder + '/prediction_fold_' + str(f))
	except Exception:
		print 'exist'
	try:
		os.mkdir(loss_folder + '/loss_fold_' + str(f))
	except Exception:
		print 'exist'
	
	fw.write('python py_new_train_hierarchical_svm.py --hier '+hier_file+' --trfeature '+data_folder+'/'+data_prefix+'_train_feature_fold_'+str(f)+' --trlabel '+data_folder+'/'+data_prefix+'_train_label_fold_'+str(f)+' --tefeature '+data_folder+'/'+data_prefix+'_test_feature_fold_'+str(f)+' --telabel '+data_folder+'/'+data_prefix+'_test_label_fold_'+str(f)+' --modelfolder '+model_folder + '/model_fold_' + str(f)+' --predictionfolder '+prediction_folder + '/prediction_fold_' + str(f) + '\n')
	fw.write('python py_compute_hierarchical_loss.py --hier '+str(hier_file)+' --telabel '+data_folder+'/'+data_prefix+'_test_label_fold_'+str(f)+'  --predictionfolder ' + prediction_folder + '/prediction_fold_' + str(f) + ' --lossf '+loss_folder + '/loss_fold_' + str(f)+'/hier_f1.txt \n')
fw.close()
os.system('chmod 777 sh_all_job.sh')
