# UserEmb_Explainable

# Model Diagram

# Runtime Platform

* OS: Ubuntu 20.04
* Python 3.8
* Java JDK 1.8
* CUDA 11
* PyTorch 1.7.1+cu110


# Data Acquisite

In this work, we have used two clinical narrative data, diabestes and MIMIC-III. Both datasets require you to take a list of training courses before acquiring the dataset.
This will ensure you follow basic ethic rules to use the datasets. The datasets should be used for ***research purposes only***.
* Diabetes
    * This dataset contains a list of candidates for clinical corhort selection. Each user/patient has multiple clinical narratives.
    * Each user/patient suffers diabetes.
    * Download 2018 (Track 1) Clinical Trial Cohort Selection Challenge: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
* MIMIC-III
    * This dataset contains ICU records from a hospital. 
    * Each user per visit will generate a list of clincal records.
    * Download data, MIMIC-III Clinical Database: https://physionet.org/content/mimiciii/1.4/

# Prerequisite

* Download and follow instructions in the [MetaMap](https://metamap.nlm.nih.gov/MetaMap.shtml) to setup MetaMap.
    * Follow the instructions to download both data (dictionaries) and metamap toolkit.
    * If you have not installed Java, please install and following the [JAVA link](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).
    * Note that our code supports both MetaMap and MetaMap-lite (faster version), but you have to change by yourself in `data_builder.py`.
* Python MetaMap Interface
    * You will install [PyMetaMap toolkit](https://github.com/AnthonyMRios/pymetamap).
    * Basically, the toolkit is a subprocess communicator between your Python script and Java files.
    * But our `data_builder.py` has its built-in supports, but you have to change codes by yourself using `metamap_concepts` or `metamaplite_concepts`.
* Other dependencies
    * Install [PyTorch](https://pytorch.org/get-started/locally/)
    * Please check the requirements.txt

# Data Analysis


# How to Run

* Data Preprocessing
    * Be sure to change data directory before running `python data_builder.py`
* Baselines
    * All baseline models will be under the directory of `./baselines/`;
    * Run `python any_baseline_script.py` will start to train user embeddings;
* Our approach
    * 

# Contact

Anonymized Version~ 


# Citation

Anonymized Version~

