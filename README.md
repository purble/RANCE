# RANCE

** Our code is built on top of the [ANCE](https://github.com/microsoft/ANCE) repository. So, please refer to it for detailed instructions regarding generating pre-processed datasets, running, and evaluating the code.

## Dataset

Download and preprocess the TREC 2019 Deep Learning (DL) Track dataset as specified in original [ANCE](https://github.com/microsoft/ANCE) repository and split the test queries into two-folds, fold1 and fold2, at random. For us, the division of the test queries resulted in the following two-folds in terms of test query ids of the TREC 2019 Deep Learning (DL) Track dataset-

*Documents Dataset*

Fold1 Test Queries : {156493, 1110199, 130510, 573724, 527433, 1037798, 1121402, 1117099, 451602, 1112341, 104861, 1132213, 1114819, 183378, 1106007, 490595, 1103812, 87452, 855410, 19335, 1129237, 146187}

Fold2 Test Queries : {1063750, 489204, 1133167, 915593, 264014, 962179, 148538, 359349, 1115776, 131843, 833860, 207786, 1124210, 287683, 87181, 443396, 1114646, 47923, 405717, 182539, 1113437}

*Passages Dataset*

Fold1 Test Queries : {156493, 168216, 1037798, 1121402, 962179, 1117099, 148538, 451602, 1115776, 104861, 207786, 1114819, 490595, 1103812, 1121709, 87452, 855410, 19335, 182539, 1113437, 1129237, 146187}

Fold2 Test Queries : {1110199, 1063750, 130510, 489204, 573724, 1133167, 527433, 915593, 264014, 359349, 1112341, 131843, 833860, 183378, 1106007, 1124210, 87181, 443396, 1114646, 47923, 405717}

We could not use a part of the training dataset as validation dataset on account of incomplete relevance judgements.

## RANCE-PRF-DEM

**Note**: The code is in the code/RANCE-PRF-DEM/ folder of this repository.
  
We have modified original [ANCE](https://github.com/microsoft/ANCE) code to sample negatives as per our proposed methodology. So, the directory structure and methodology to run the code remains the same, except that during each ANN-data generation step we sample negatives with the help of one of the folds as validation dataset and evluate the trained model on the other fold. The final scores are obtained by averaging the performance of two models evaluated on different test folds. We mainly modified the sampling strategy in the *code/RANCE-PRFS-DEM/drivers/run_ann_data_gen.py* file.

The tables below provides results of the two models trained on each of the folds and then evaluated on the other fold. Hyperlinks embedding in the header of the tables can be used to download our trained models.

*Passage*

|             |               | [Model_Fold1dev_Fold2test](https://github.com/microsoft/ANCE)  | [Model_Fold1test_Fold2dev](https://github.com/microsoft/ANCE)  | Average Performance |
|-------------|---------------|----------------------------|-----------------------------|---------------------|
| *Re-Rerank* | NDCG          |          0.672             |          0.689              |        0.681        |
|             | Recall        |          0.619             |          0.734              |        0.676        |
|             | MRR           |          1.0               |          0.932              |        0.966        |
| *Retreival* | NDCG          |          0.661             |          0.667              |        0.664        |
|             | Recall        |          0.621             |          0.728              |        0.674        |
|             | MRR           |          0.931             |          0.939              |        0.935        |


*Document*

|             |               | [Model_Fold1dev_Fold2test](https://github.com/microsoft/ANCE)  | [Model_Fold1test_Fold2dev](https://github.com/microsoft/ANCE)  | Average Performance |
|-------------|---------------|----------------------------|-----------------------------|---------------------|
| *Re-Rerank* | NDCG          |          0.704             |          0.655              |        0.68         |
|             | Recall        |          0.334             |          0.297              |        0.315        |
|             | MRR           |          0.922             |          0.918              |        0.92         |
| *Retreival* | NDCG          |          0.652             |          0.632              |        0.642        |
|             | Recall        |          0.295             |          0.293              |        0.294        |
|             | MRR           |          0.921             |          0.913              |        0.917        |


## RANCE-PRF and RANCE

**Note**: The code is in the code/RANCE-PRF/ folder of this repository.

**RANCE-PRF**

For DEM implementation on top of [ANCE](https://github.com/microsoft/ANCE) code, we have made the following modifications-

* We initially generated corpus level statistics like term frequency, document lengths, document frequency etc., required to compute BM25 score for input queries.
* We added a BM25_helper object to *code/RANCE-PRF/utils/utils.py* that essentially loads these statistics at the start of the execution of *code/RANCE-PRF/utils/run_ann_data_gen.py* script.
* In addition to sampling negatives as per our proposed strategy as employed in RANCE-PRF-DEM, we also compute the BM25 score for each training query and save it as a part of the updated dataset generated by *code/RANCE-PRF/utils/run_ann_data_gen.py* script for each new checkpoint.
* We have modified the loss function formulation in *code/RANCE-PRF/utils/run_ann.py* as DEM strategy.

**RANCE**

We obtain final scores for our proposed RANCE method using a model trained using RANCE-PRF strategy, and add PRF during evaluation for both re-ranking and retrieval tasks.

The tables below provides results of the two models trained on each of the folds using RANCE-PRF strategy and then evaluated on the other fold after incorporating PRF. Hyperlinks embedding in the header of the tables can be used to download our trained models.

*Passage*

|             |               | [Model_Fold1dev_Fold2test](https://github.com/microsoft/ANCE)  | [Model_Fold1test_Fold2dev](https://github.com/microsoft/ANCE)  | Average Performance |
|-------------|---------------|----------------------------|-----------------------------|---------------------|
| *Re-Rerank* | NDCG          |          0.708             |          0.696              |        0.702        |
|             | Recall        |          0.734             |          0.619              |        0.676        |
|             | MRR           |          0.930             |          0.976              |        0.954        |
| *Retreival* | NDCG          |          0.690             |          0.700              |        0.695        |
|             | Recall        |          0.767             |          0.626              |        0.697        |
|             | MRR           |          0.878             |          1.0                |        0.939        |


*Document*

|             |               | [Model_Fold1dev_Fold2test](https://github.com/microsoft/ANCE)  | [Model_Fold1test_Fold2dev](https://github.com/microsoft/ANCE)  | Average Performance |
|-------------|---------------|----------------------------|-----------------------------|---------------------|
| *Re-Rerank* | NDCG          |          0.699             |          0.704              |        0.702        |
|             | Recall        |          0.299             |          0.350              |        0.325        |
|             | MRR           |          0.915             |          0.901              |        0.908        |
| *Retreival* | NDCG          |          0.663             |          0.691              |        0.677        |
|             | Recall        |          0.320             |          0.293              |        0.307        |
|             | MRR           |          0.911             |          0.892              |        0.901        |
