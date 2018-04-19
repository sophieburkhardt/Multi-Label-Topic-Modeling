This repository contains classifiers and inference/evaluation methods for the paper Online Multi-Label Dependency Topic Models for Text Classification. The paper will be published in the Machine Learning Journal.

Included classifiers are:

- BR (1. modified from Meka, 2. a python script using LIBLINEAR for the SVMs)
- SCVB
- SCVB-Dependency
- Dependency-LDA
- Fast-Dependency-LDA (the reversed version corresponds to the author-topic model)
 
 
 to build the project run `mvn package`
 to install Mallet in your maven repository run 
 `mvn install:install-file -Dfile=jars/mallet-2.0.9-SNAPSHOT.jar -DpomFile=jars/mallet-2.0.9-SNAPSHOT.pom`
 to run a classifier, you need to specify options in the `config.txt` file, e.g. for Dependency-LDA:
```
 --dataFilename data/datafile --testDataFilename data/datafile --datasetName name --numIterations 100 --numTopWords 20 --wordThreshold 5 --batch --dependency --alphaSum 0 --testAlphaSum 30 --beta 0.01 --beta_c 0.01 --gamma 10 --testGamma 10 --eta 100 --testEta 100 --innerIterations 1 --t 100 --supervised true --trainingPercentage 1.0 --evaluationInterval 100 --writeDistributions false --print false --run 1
 ```
 
 You may then run the program with `java -jar target/Multi-Label-Topic-Modeling-1.0-SNAPSHOT.jar linenumber` where linenumber is the line in the config.txt file that you want to use
 
General options are:

- **batch** run classifier in batch mode with a train and a test set
- **incremental** run classifier in incremental mode where the classifier is first trained on an initial batch and then alternately tested on the next batch before being trained on the next batch (only available for SCVB and SCVB-Dependency)
  - additional parameter to specify: --batchSize (e.g. batchSize 100 means train on 100 instances then test on the next 100 instances etc.)
- **dataFilename** the filename with the training data
- **testDataFilename** the filename with the testing data
- **datasetName** the name to use for the path to the results file
- **writeDistributions** true or false, whether or not to write the distributions for the testdataset to file, might be needed for later averaging, takes up a lot of memory depending on the size of the testset and the number of labels
- **print** true or false, whether or not to write the resulting word clouds to file
  - further options: **numTopWords** and **wordThreshold** for the number of words per label and the frequency threshold for writing words
- **evaluationInterval** after how many iterations the classifier is tested on the testset
- **run** an integer may be specified that is included in the filename to separate multiple runs with the same configuration
- **supervised** currently always has to be set to true
 
 
 The options for the different methods are:
 
- Dependency-LDA, Fast-Dependency-LDA: 
  - **dependency** for using Dependency-LDA
  - **fastdependency** for using Fast-Dependency-LDA
  - **s2** set to true if you want to use the evaluation strategy S2
  - **alphaSum** for training
  - **testAlphaSum** for testing
  - **beta**
  - **beta_c** corresponds to beta_y in the paper
  - **t** number of topics to use
  - **gamma** for training (not for Fast-Dependency-LDA)
  - **testGamma** for testing (not for Fast-Dependency-LDA)
  - **eta** for calculation of alpha' during training (not for Fast-Dependency-LDA)
  - **testEta**  for calculation of alpha' during testing (not for Fast-Dependency-LDA)
  - **innerIterations** currently not implemented, use a value of 1 (not for Fast-Dependency-LDA)
- SCVB, SCVB-Dependency:
  - **SCVB** for using SCVB
  - **SCVBDependency** for using SCVB-Dependency
  - **alpha**
  - **alphaDep** not used
  - **eta** corresponds to beta
  - **etaDep** corresponds to beta_y
  - **rho** update parameter (set to 0.9)
  - **rhophi** update parameter (set to 0.9)
  

The format of the input file is as follows:
Each line is one input document. There are three columns expected, separated by Tabs

1. ID of the document
2. one or more labels, separated by white spaces
3. the text of the document itself

