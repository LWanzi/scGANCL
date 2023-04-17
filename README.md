# GANCL: a self-supervised contrastive learning imputation method for scRNA-seq data with generative adversarial network


![model](https://github.com/LWanzi/GANCL/blob/origin/GANCL.png)

Introduction
-----

we propose a novel self-supervised deep learning model called GANCL for scRNA-seq data imputation. GANCL incorporates generative adversarial network(GAN) and contrastive learning(CL), which enhances model training capabilities through stronger data augmentation. GANCL can learn discriminative representation by distinguishing real samples from generated samples. Moreover, we introduce a zero-inflated negative binomial(ZINB) distribution in the GAN framework to model the original probability distribution of scRNA-seq data. We evaluate GANCL on various downstream analysis tasks, including clustering, gene expression recovery, trajectory inference, and differentially expressed gene (DEG) analysis. The results showed that GANCL outperforms four state-of-the-art methods consistently. Besides, ablation studies were conducted to demonstrate the contributions of each component of GANCL.

Requirement
-----
Python == 3.6.13

Pytorch==1.10.0

h5py==3.1.0

scanpy==1.7.2

umap-learn == 0.5.3

Usage
-----
You can run the GANCL from the command line:

$ python main.py --dataset adam --epochs 200

Arguments
-----

|    Parameter    | Introduction                                                 |
| :-------------: | ------------------------------------------------------------ |
|    dataset     | A h5 file. Contains a matrix of scRNA-seq expression values,true labels, and other information. By default, genes are assumed to be represented by rows and samples are assumed to be represented by columns.|
|    task     |Downstream task, default is clustering |
|     epochs     | Number of training epochs                                    |
