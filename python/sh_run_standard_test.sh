#test ohloh
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/ohloh/train_sourceforge_ohloh_standard_feature.svm --trlabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --tefeature /home/xiao/workspace/enrich/ohloh/test_sourceforge_ohloh_standard_feature.svm --telabel /home/xiao/workspace/enrich/ohloh/test_labels.txt --modelfolder /home/xiao/workspace/enrich/sf_oh_models --predictionfolder /home/xiao/workspace/enrich/sf_oh_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/ohloh/test_labels.txt  --predictionfolder /home/xiao/workspace/enrich/sf_oh_prediction --lossfolder /home/xiao/workspace/enrich/sf_oh_loss/hier_f1.txt 
#test freecode
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/freecode/train_sourceforge_freecode_standard_feature.svm --trlabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --tefeature /home/xiao/workspace/enrich/freecode/test_sourceforge_freecode_standard_feature.svm --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt --modelfolder /home/xiao/workspace/enrich/sf_fr_models --predictionfolder /home/xiao/workspace/enrich/sf_fr_prediction
#test on the test set
python py_new_predict_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --tefeature /home/xiao/workspace/enrich/freecode/test_sourceforge_freecode_standard_feature.svm --modelfolder /home/xiao/workspace/enrich/sf_fr_models --predictionfolder /home/xiao/workspace/enrich/sf_fr_test_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt  --predictionfolder /home/xiao/workspace/enrich/sf_fr_test_prediction --lossfolder /home/xiao/workspace/enrich/sf_fr_loss/hier_f1_test.txt 
#test on the trainig set
python py_new_predict_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --tefeature /home/xiao/workspace/enrich/freecode/train_sourceforge_freecode_standard_feature.svm --modelfolder /home/xiao/workspace/enrich/sf_fr_models --predictionfolder /home/xiao/workspace/enrich/sf_fr_train_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --predictionfolder /home/xiao/workspace/enrich/sf_fr_train_prediction --lossfolder /home/xiao/workspace/enrich/sf_fr_loss/hier_f1_train.txt 
#conver the predicted probs into feature
python py_generate_meta_features_from_probs.py --folder /home/xiao/workspace/enrich/sf_fr_train_prediction --ext test_probs --outsvm /home/xiao/workspace/enrich/freecode/train_meta_sourceforge_freecode_standard_feature.svm
python py_generate_meta_features_from_probs.py --folder /home/xiao/workspace/enrich/sf_fr_test_prediction --ext test_probs --outsvm /home/xiao/workspace/enrich/freecode/test_meta_sourceforge_freecode_standard_feature.svm
#train the model on meta feature and test it again
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/freecode/train_meta_sourceforge_freecode_standard_feature.svm --trlabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --tefeature /home/xiao/workspace/enrich/freecode/test_meta_sourceforge_freecode_standard_feature.svm --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt --modelfolder /home/xiao/workspace/enrich/sf_fr_meta_models --predictionfolder /home/xiao/workspace/enrich/sf_fr_meta_test_prediction
python py_new_predict_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --tefeature /home/xiao/workspace/enrich/freecode/train_meta_sourceforge_freecode_standard_feature.svm --modelfolder /home/xiao/workspace/enrich/sf_fr_meta_models --predictionfolder /home/xiao/workspace/enrich/sf_fr_meta_train_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --predictionfolder /home/xiao/workspace/enrich/sf_fr_meta_train_prediction --lossfolder /home/xiao/workspace/enrich/sf_fr_meta_train_loss/hier_f1_train.txt 
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt --predictionfolder /home/xiao/workspace/enrich/sf_fr_meta_test_prediction --lossfolder /home/xiao/workspace/enrich/sf_fr_meta_test_loss/hier_f1_test.txt 

######################
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt  --predictionfolder /home/xiao/workspace/enrich/sf_fr_prediction --lossfolder /home/xiao/workspace/enrich/sf_fr_loss/hier_f1.txt 
python py_new_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt  --predictionfolder /home/xiao/workspace/enrich/sf_fr_prediction --lossfolder /home/xiao/workspace/enrich/sf_fr_loss/hier_f1_max_prob.txt 
#sourceforge on free code with TF
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/freecode/train_sourceforge_freecode_standard_feature_tf.svm --trlabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --tefeature /home/xiao/workspace/enrich/freecode/test_sourceforge_freecode_standard_feature_tf.svm --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt --modelfolder /home/xiao/workspace/enrich/sf_fr_models --predictionfolder /home/xiao/workspace/enrich/sf_fr_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt  --predictionfolder /home/xiao/workspace/enrich/sf_fr_prediction --lossfolder /home/xiao/workspace/enrich/sf_fr_loss/hier_f1_tf.txt 
python py_new_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt  --predictionfolder /home/xiao/workspace/enrich/sf_fr_prediction --lossfolder /home/xiao/workspace/enrich/sf_fr_loss/hier_f1_max_prob_tf.txt 
#test sourceforge with common name with freecode
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/sourceforge/train_fc_sourceforge_sourceforge_standard_feature.svm --trlabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --tefeature /home/xiao/workspace/enrich/sourceforge/test_fc_sourceforge_sourceforge_standard_feature.svm --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt --modelfolder /home/xiao/workspace/enrich/sf_sf_fr_models --predictionfolder /home/xiao/workspace/enrich/sf_sf_fr_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt  --predictionfolder /home/xiao/workspace/enrich/sf_sf_fr_prediction --lossfolder /home/xiao/workspace/enrich/sf_sf_fr_loss/hier_f1.txt 
#test sourceforge with common name with ohloh
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/sourceforge/train_oh_sourceforge_sourceforge_standard_feature.svm --trlabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --tefeature /home/xiao/workspace/enrich/sourceforge/test_oh_sourceforge_sourceforge_standard_feature.svm --telabel /home/xiao/workspace/enrich/ohloh/test_labels.txt --modelfolder /home/xiao/workspace/enrich/sf_sf_oh_models --predictionfolder /home/xiao/workspace/enrich/sf_sf_oh_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/ohloh/test_labels.txt  --predictionfolder /home/xiao/workspace/enrich/sf_sf_oh_prediction --lossfolder /home/xiao/workspace/enrich/sf_sf_oh_loss/hier_f1.txt 

#test sourceforege on sourceforge training set
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/sourceforge/Train_sourceforge_standard_feature.svm --trlabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --tefeature /home/xiao/workspace/enrich/sourceforge/Train_sourceforge_standard_feature.svm --telabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --modelfolder /home/xiao/workspace/enrich/sf_models --predictionfolder /home/xiao/workspace/enrich/sf_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt --predictionfolder /home/xiao/workspace/enrich/sf_prediction --lossfolder /home/xiao/workspace/enrich/sf_loss/hier_f1.txt 

#test freecode on freecode training set
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/freecode/freecode_freecode_feature.svm --trlabel /home/xiao/workspace/enrich/freecode/test_labels.txt --tefeature /home/xiao/workspace/enrich/freecode/freecode_freecode_feature.svm --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt --modelfolder /home/xiao/workspace/enrich/fc_models --predictionfolder /home/xiao/workspace/enrich/fc_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt --predictionfolder /home/xiao/workspace/enrich/fc_prediction --lossfolder /home/xiao/workspace/enrich/fc_loss/hier_f1.txt 

#test ohloh on ohloh training set
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/ohloh/ohloh_ohloh_feature.svm --trlabel /home/xiao/workspace/enrich/ohloh/test_labels.txt --tefeature /home/xiao/workspace/enrich/ohloh/ohloh_ohloh_feature.svm --telabel /home/xiao/workspace/enrich/ohloh/test_labels.txt --modelfolder /home/xiao/workspace/enrich/oh_models --predictionfolder /home/xiao/workspace/enrich/oh_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/ohloh/test_labels.txt --predictionfolder /home/xiao/workspace/enrich/oh_prediction --lossfolder /home/xiao/workspace/enrich/oh_loss/hier_f1.txt 


#new Dec 10, 2012. Test with example less than 5 labels
#test freecode
python py_new_train_hierarchical_svm.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --trfeature /home/xiao/workspace/enrich/freecode/train_sourceforge_freecode_standard_feature.svm_5 --trlabel /home/xiao/workspace/enrich/sourceforge/sf_labels.txt_5 --tefeature /home/xiao/workspace/enrich/sourceforge/test_fc_sourceforge_sourceforge_standard_feature.svm_5 --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt_5 --modelfolder /home/xiao/workspace/enrich/tmp_models --predictionfolder /home/xiao/workspace/enrich/tmp_prediction
python py_compute_hierarchical_loss.py --hier /home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt --telabel /home/xiao/workspace/enrich/freecode/test_labels.txt_5  --predictionfolder /home/xiao/workspace/enrich/tmp_prediction --lossfolder /home/xiao/workspace/enrich/tmp_loss/



#test 20 news
#train and test
python py_old_train_hierarchical_svm.py --hier /home/mpi/topic_model_svm/20_news/20_news_parent2children_hierarchy.txt --trfeature /home/mpi/topic_model_svm/20_news/20_news_0_fold_train_text.svm --trlabel /home/mpi/topic_model_svm/20_news/20_news_0_fold_train_label --tefeature /home/mpi/topic_model_svm/20_news/20_news_0_fold_test_text.svm --telabel /home/mpi/topic_model_svm/20_news/20_news_0_fold_test_label --modelfolder /home/xiao/liblinear-1.92_modified_xiao/python/tmp_model --predictionfolder /home/xiao/liblinear-1.92_modified_xiao/python/tmp_prediction --trainpredictionfolder train_prediction
#make predicted label and compute loss
python py_compute_hierarchical_loss.py --hier /home/mpi/topic_model_svm/20_news/20_news_parent2children_hierarchy.txt --telabel /home/mpi/topic_model_svm/20_news/20_news_0_fold_test_label  --predictionfolder /home/xiao/liblinear-1.92_modified_xiao/python/tmp_prediction --lossfolder /home/xiao/liblinear-1.92_modified_xiao/python/tmp_loss
/home/mpi/topic_model_svm/ComputeHierLabels --t 0.5 FILES /home/mpi/topic_model_svm/20_news/20_news_hierarchy.txt tmp_prediction test_probs /home/mpi/topic_model_svm/20_news/20_news_0_fold_test_label fold_0  tmp_loss /home/xiao/liblinear-1.92_modified_xiao/python
#compute C++ loss
/home/mpi/topic_model_svm/ComputeHierLoss --t 0.5 FILES /home/mpi/topic_model_svm/20_news/20_news_hierarchy.txt tmp_prediction test_probs /home/mpi/topic_model_svm/20_news/20_news_0_fold_test_label fold_0  tmp_loss /home/xiao/liblinear-1.92_modified_xiao/python



#with new method
python py_new_train_hierarchical_svm.py --hier /home/mpi/topic_model_svm/20_news/20_news_parent2children_hierarchy.txt --trfeature /home/mpi/topic_model_svm/20_news/20_news_0_fold_train_text.svm --trlabel /home/mpi/topic_model_svm/20_news/20_news_0_fold_train_label --tefeature /home/mpi/topic_model_svm/20_news/20_news_0_fold_test_text.svm --telabel /home/mpi/topic_model_svm/20_news/20_news_0_fold_test_label --modelfolder /home/xiao/liblinear-1.92_modified_xiao/python/new_tmp_model --predictionfolder /home/xiao/liblinear-1.92_modified_xiao/python/new_tmp_prediction --trainpredictionfolder new_train_prediction
#make predicted label and compute loss
/home/mpi/topic_model_svm/ComputeHierLabels --t 0.5 FILES /home/mpi/topic_model_svm/20_news/20_news_hierarchy.txt new_tmp_prediction test_probs /home/mpi/topic_model_svm/20_news/20_news_0_fold_test_label fold_0  new_tmp_loss /home/xiao/liblinear-1.92_modified_xiao/python
/home/mpi/topic_model_svm/ComputeHierLoss --t 0.5 FILES /home/mpi/topic_model_svm/20_news/20_news_hierarchy.txt new_tmp_prediction test_probs /home/mpi/topic_model_svm/20_news/20_news_0_fold_test_label fold_0  new_tmp_loss /home/xiao/liblinear-1.92_modified_xiao/python