raw_data_dir=
preprocessed_data_dir=

python msmarco_data.py 
--data_dir $raw_data_dir \
--out_data_dir $preprocessed_data_dir \ 
--model_type use rdot_nll \ 
--model_name_or_path roberta-base \ 
--max_seq_length 512 \ 
--data_type 1

