python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/freecode/train_sourceforge_freecode_standard_feature.svm --trlabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --tefeature /home/xiao/workspace/enrich/freecode/test_sourceforge_freecode_standard_feature.svm --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt --modelfolder /home/xiao/workspace/enrich/freecode/model_svm --predictionfolder /home/xiao/workspace/enrich/freecode/prediction_svm
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt  --predictionfolder /home/xiao/workspace/enrich/freecode/prediction_svm


python py_new_predict_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --tefeature /home/xiao/workspace/enrich/freecode/train_sourceforge_freecode_standard_feature.svm --modelfolder /home/xiao/workspace/enrich/freecode/model_svm --predictionfolder /home/xiao/workspace/enrich/freecode/prediction_svm_train
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --predictionfolder /home/xiao/workspace/enrich/freecode/prediction_svm_train


python py_new_predict_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --tefeature /home/xiao/workspace/enrich/freecode/test_sourceforge_freecode_standard_feature.svm --modelfolder /home/xiao/workspace/enrich/freecode/model_svm --predictionfolder /home/xiao/workspace/enrich/freecode/prediction_svm
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt --predictionfolder /home/xiao/workspace/enrich/freecode/prediction_svm
