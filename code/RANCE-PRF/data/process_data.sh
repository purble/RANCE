raw_data_dir=/home/prafullpraka/scratch/mst/ANCE/data/raw_data
preprocessed_data_dir=/home/prafullpraka/scratch/mst/ANCE/data/preprocessed_data/document/roberta_base/firstp

python msmarco_data.py 
--data_dir $raw_data_dir \
--out_data_dir $preprocessed_data_dir \ 
--model_type use rdot_nll \ 
--model_name_or_path roberta-base \ 
--max_seq_length 512 \ 
--data_type 1

