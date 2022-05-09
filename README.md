# spacy_mi
Membership inference attacks on spacy language models

Currently membership inference attack on using a full data is performed via spacy3_mi_full_dataset_MI_attack.py. That file is run via a bash file that passes the following arguments as input.

``
#$1 - epoch
#$2 - attach_type
#$3 - batch_size
#$4 - train_set
#$5 - member_set
#$6 - non_member_set
#$7 - subruns
#$8 - label
#$9 - test_set
``

`bash spacy3_full_dataset_MI_attack.sh 75 MIA 50 dataset/i2b2_n2c2_train_data_ID_size_more_than_0_count_None.pickle dataset/i2b2_n2c2_ID_size_more_than_0_count_None_member_sentences.pickle dataset/i2b2_n2c2_ID_size_more_than_0_count_None_non_member_sentences.pickle 10 "ID" dataset/i2b2_n2c2_test_data_ID_size_more_than_0_count_None.pickle`

Breaking down the above command:
`epoch = 50`
`attach_type = MIA`
`batch_size = 50`
`train_set = dataset/i2b2_n2c2_train_data_ID_size_more_than_0_count_None.pickle`
`member_set = dataset/i2b2_n2c2_ID_size_more_than_0_count_None_member_sentences.pickle`
`non_member_set = dataset/i2b2_n2c2_ID_size_more_than_0_count_None_non_member_sentences.pickle`
`subruns = 10`
`label = "ID"`
`test_set = dataset/i2b2_n2c2_test_data_ID_size_more_than_0_count_None.pickle`

We train on the i2b2_n2c2 dataset for the ID entity. size_more_than_0 means taking ID with more than 0 characters.
This dataset is split into train and test set with member set including ID in the training data and non_member set not including ID.
test_set is used to get metrics for MIA.

The output is stored in the results folder and would be stored in a folder structured as:
`20210908_spacy_3.1.2_attack_MIA_epochs_75_batch_size_dataset_i2b2_n2c2_train_data_ID_size_more_than_0_count_None`



