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

## RANCE-PRFS-DEM

**Note**: The code is in the code/RANCE-PRFS-DEM/ folder of this repository.
  
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


## RANCE-PRFS and RANCE


