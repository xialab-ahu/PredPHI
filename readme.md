## **PredPHI**

### Title

A deep learning-based method for identification of bacteriophage-host interaction

### Developers

Menglu Li (mengluli@foxmail.com), Yannan Bin and Junfeng Xia (jfxia@ahu.edu.cn) from School of Computer Science and Technology, Institutes of Physical Science and Information Technology, Anhui University.

Yanan Wang and Fuyi Li from Biomedicine Discovery Institute and Department of Biochemistry & Molecular Biology, Monash Centre for Data Science, Monash University.

Yun Zhao and Jian Li from Biomedicine Discovery Institute and Department of Biochemistry & Molecular Biology,  Monash University.

Mengya Liu and Sijia Zhang from Institutes of Physical Science and Information Technology, Anhui University.

Geoffrey I. Webb from Monash Centre for Data Science, Monash University.

Jiangning Song ( Jiangning.Song@monash.edu) from Biomedicine Discovery Institute and Department of Biochemistry & Molecular Biology, Monash Centre for Data Science, ARC Centre of Excellence in Advanced Molecular Imaging, Monash University.

### Related Files

#### data

| FILE NAME           | DESCRIPTION                                                  |
| :------------------ | :----------------------------------------------------------- |
| training_set.csv    | the data used to train model (include phage name, host name, and class) |
| training_kmeans.csv | the data used to train model (use K-Means clustering method to select negative samples, construct balanced training set) |
| test_set.csv        | the data used to test model (include phage name, host name, and class) |
| test_kmeans.csv     | the data used to test model (use K-Means clustering method to select negative samples, construct balanced test set) |
| test-random.csv     | the data used to test model (randomly select negative samples to balance test set) |
| min_num.csv         | minimum feature file (for normalizing new feature)           |
| max_num.csv         | maximum feature file (for normalizing new feature)           |
| mediumdata          | save medium files when run codes                             |
| trainingfeatures    | save training features (file name is phage and host name)    |
| testfeatures        | save test features (file name is phage and host name)        |
| test-test.csv       | the data used to test 1-obtainfeatures.py code (include phage name, host name, and class) |
| test-test-seq.fasta | the protein sequence encoded by phage and host in test-test.csv |

#### code

| FILE NAME           | DESCRIPTION                                                  |
| :------------------ | :----------------------------------------------------------- |
| 1-obtainfeatures.py | obtain phage and host features (the result save in trainingfeatures and testfeatures) |
| 2-training-model.py | train model                                                  |
| 3-test-result.py    | test model result                                            |

#### result

| FILE NAME | DESCRIPTION                                       |
| --------- | ------------------------------------------------- |
| model.h5  | the trained model can be directly used to predict |

### Contact

Please feel free to contact us if you need any help.

