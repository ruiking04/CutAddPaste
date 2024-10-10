# CutAddPaste: Time Series Anomaly Detection by Exploiting Abnormal Knowledge
This repository provides the implementation of the _CutAddPaste: Time Series Anomaly Detection by Exploiting Abnormal Knowledge_ method, called _CutAddPaste_ below. 

## Abstract
> Detecting time-series anomalies is extremely intricate due to the rarity of anomalies and imbalanced sample categories, 
> which often result in costly and challenging anomaly labeling. 
> Most of the existing approaches largely depend on assumptions of normality, overlooking labeled abnormal samples. 
> While methods based on anomaly assumptions can incorporate prior knowledge of anomalies for data augmentation in training classifiers, 
> the adopted random or coarse-grained augmentations solely focus on point-wise anomalies and lack cutting-edge domain knowledge, 
> making them less likely to achieve better performance.
> This paper introduces CutAddPaste, a novel anomaly assumption-based approach for detecting time-series anomalies. 
> It primarily employs a data augmentation strategy to generate pseudo anomalies, 
> by exploiting prior knowledge of anomalies as much as possible. At the core of "CutAddPaste" is cutting patches from 
> random positions in temporal subsequence samples, adding linear trend terms, and pasting them into other samples, 
> so that it can well approximate a variety of anomalies, including point and pattern anomalies. 
> Experiments on standard benchmark datasets demonstrate that our method outperforms the state-of-the-art approaches.



## Citation
Link to our paper [here](https://dl.acm.org/doi/10.1145/3637528.3671739).
If you use this code for your research, please cite our paper:

```
@inproceedings{10.1145/3637528.3671739,
author = {Wang, Rui and Mou, Xudong and Yang, Renyu and Gao, Kai and Liu, Pin and Liu, Chongwei and Wo, Tianyu and Liu, Xudong},
title = {CutAddPaste: Time Series Anomaly Detection by Exploiting Abnormal Knowledge},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671739},
doi = {10.1145/3637528.3671739},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {3176–3187},
numpages = {12},
keywords = {abnormal knowledge, anomaly detection, anomaly-assumption, data augmentation, time series},
location = {Barcelona, Spain},
series = {KDD '24}
}
```

## Installation
This code is based on `Python 3.6`, all requirements are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as Merlion suggested.

```
pip install salesforce-merlion==1.1.1
pip install -r requirements.txt
```

## Dataset
We acknowledge the contributors of the dataset, including AIOps, UCR, SWaT, and WADI.
This repository already includes Merlion's data loading package `ts_datasets`.

### AIOps (KPI, IOpsCompetition) and UCR. 
1. AIOps Link: https://github.com/NetManAIOps/KPI-Anomaly-Detection
2. UCR Link: https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ 
and https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
3. Download and unzip the data in `data/iops_competition` and `data/ucr` respectively. 
e.g. For AIOps, download `phase2.zip` and unzip the `data/iops_competition/phase2.zip` before running the program.

### SWaT and WADI. 
1. For SWaT and WADI, you need to apply by their official tutorial. Link: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
2. Because multiple versions of these two datasets exist, 
we used their newer versions: `SWaT.SWaT.A2_Dec2015, version 0` and `WADI.A2_19Nov2019`.
3. Download and unzip the data in `data/swat` and `data/wadi` respectively. Then run the 
`swat_preprocessing()` and `wadi_preprocessing()` functions in `dataloader/data_preprocessing.py` for preprocessing.


## Repository Structure

### `conf`
This directory contains experiment parameters for all models on AIOps (as `IOpsCompetition` in the code), UCR, SWaT, and WADI datasets.

### `models`
Source code of CutAddPaste model.

### `results`
Directory where the experiment result is saved.

## CutAddPaste Usage
```
# CutAddPaste Method (dataset_name: IOpsCompetition, UCR, SWaT, WADI)
python cutAddPaste.py --selected_dataset <dataset_name> --device cuda --seed 2
```

## Baselines
Anomaly Transformer(AOT), COCA, AOC, RandomScore(RAS), InputOrigin(AAS), NCAD, LSTMED, OC_SVM, ISF, SR, RRCF, SVDD, DAMP, TS_AD(TCC)

We reiterate that in addition to our method, the source code of other baselines is based on the GitHub source code 
provided by their papers. For reproducibility, we changed the source code of their models as little as possible. 
We are grateful for the work on these papers.

We consult the GitHub source code of the paper corresponding to the baseline and then reproduce it. 
For baselines that use the same datasets as ours, we use their own recommended hyperparameters. 
For different datasets, we use the same hyperparameter optimization method Grid Search as our model to find the optimal hyperparameters.

### Acknowledgements
Part of the code, especially the baseline code, is based on the following source code.
1. [Anomaly Transformer(AOT)](https://github.com/thuml/Anomaly-Transformer)
2. [COCA](https://github.com/ruiking04/COCA)
3. [AOC](https://github.com/alsike22/AOC)
4. [Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch)
5. [TS-TCC](https://github.com/emadeldeen24/TS-TCC)
6. [DAMP](https://sites.google.com/view/discord-aware-matrix-profile/documentation) and 
[DAMP-python](https://github.com/sihohan/DAMP)
7. LSTM_ED, SR, and IF are reproduced based on [saleforce-merlion](https://github.com/salesforce/Merlion/tree/main/merlion/models/anomaly)
8. [RRCF](https://github.com/kLabUM/rrcf?tab=readme-ov-file)
9. [Metrics:affiliation-metrics](https://github.com/ahstat/affiliation-metrics-py)

