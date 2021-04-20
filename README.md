# RANCE

## Dataset

Download and preprocess the TREC 2019 Deep Learning (DL) Track dataset as specified in original [ANCE](https://github.com/microsoft/ANCE) repository and split the test queries into two-folds, fold1 and fold2, at random. For us, the division of the test queries resulted in the following two-folds in terms of test query ids of the TREC 2019 Deep Learning (DL) Track dataset-

*Documents Dataset*

Fold1 Test Queries : {156493, 1110199, 130510, 573724, 527433, 1037798, 1121402, 1117099, 451602, 1112341, 104861, 1132213, 1114819, 183378, 1106007, 490595, 1103812, 87452, 855410, 19335, 1129237, 146187}

Fold2 Test Queries : {1063750, 489204, 1133167, 915593, 264014, 962179, 148538, 359349, 1115776, 131843, 833860, 207786, 1124210, 287683, 87181, 443396, 1114646, 47923, 405717, 182539, 1113437}

*Passages Dataset*

Fold1 Test Queries : {156493, 168216, 1037798, 1121402, 962179, 1117099, 148538, 451602, 1115776, 104861, 207786, 1114819, 490595, 1103812, 1121709, 87452, 855410, 19335, 182539, 1113437, 1129237, 146187}

Fold2 Test Queries : {1110199, 1063750, 130510, 489204, 573724, 1133167, 527433, 915593, 264014, 359349, 1112341, 131843, 833860, 183378, 1106007, 1124210, 87181, 443396, 1114646, 47923, 405717}

We could not use a part of the training dataset as validation dataset on account of incomplete relevance judgements.

## RANCE-PRFS-DEM

<u>Note<\u>: Switch to RANCE-PRFS-DEM branch of this repository for the code.
  
We have modified original [ANCE](https://github.com/microsoft/ANCE) code to sample negatives as per our proposed methodology. So, the directory structure and methodology to run the code remains the same, except that during each ANN-data generation step we sample negatives with the help of one of the folds as validation dataset and evluate the trained model on the other fold. The final scores are obtained by averaging the performance of two models evaluated on different test folds.

The table below provides results of the two models trained on each of the folds and then evaluated on the other folds. The header of the table has an embedded hyperlink that can be used to download our trained models.

|-----------| ------------- | ---------------------------------------------- | ---------------------------------------------- |
|           |               | Model Trained Using Fold1 & Evaluated on Fold2 | Model Trained Using Fold2 & Evaluated on Fold1 |
|-----------| ------------- | ---------------------------------------------- | ---------------------------------------------- |
| Re-Rerank | NDCG@10       |                                                |                                                |
|           | MRR           |                                                |                                                |
|           | Recall        |                                                |                                                |
|-----------| ------------- | ---------------------------------------------- | ---------------------------------------------- |
| Retreival | NDCG@10       |                                                |                                                |
|           | MRR           |                                                |                                                |
|           | Recall        |                                                |                                                |
|-----------| ------------- | ---------------------------------------------- | ---------------------------------------------- |

## RANCE-PRFS

## RANCE

