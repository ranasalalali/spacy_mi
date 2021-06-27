# spacy_mi
Membership inference attacks on spacy language models

## tham_branch is for MIA based on timing side channel

### (0) Generate password

Run this script "generate_password.py", can choose parameter for number of digit, letter, symbol etc


### (1) Update NER model:

First, we update the original spacy ner model with a list of generated password from this list: "passwords_list_5000_min_lower_1_min_upper_1_min_digit_1_min_spec_1_min_len_6"

Due to the large size of this updated model, you should update the model locally. Run the script updating_NER_model_en_core_web_lg.py

The updated model: "updated_ner_with_2000_password_min_1_1_1_1_6".
It is saved for later use in launching attacks.


### (2) Launching MIA attacks


Run this script for both attacks (original NER and updated NER model): "test_updated_ner_with_VM.py"


When launching attack, we first query the original model with a list of in-vocab word (from orginal vocab) and a list of generated password. 

We also query the updated ner model with a list of generated passwords (which were used to update the model) and a list of newly generated password.


Results are in folder: ./password_11116_updated_ner_ROC_20210505 where there are pickle files for both attacks original ner model and updated ner model, runing on VM or Tham's local PC.
Note the results in paper were based on experiments run on Tham's local PC (ubuntu 18, RAM 16 GB)

### Visuallization of results

Run this script: plot_roc_NER.py

expeirmental results are in folder ./password_11116_updated_ner_ROC_20210505

Graphs are saved to ./new_plot_ner_PC_roc_3.0.3/

Graphs used in ccs paper version: 

1) new1_absolute_time_run1_orig_ner_VM_3.0.3.pdf
2) new1_absolute_time_run1_updated_ner_VM_3.0.3.pdf
3) roc_time_abs_both_task_tok_ner_2000_words_PC_both_ner_3.0.3.pdf

