# scGANCL: bidirectional generative adversarial network for imputing scRNA-seq data with contrastive learning


![model](https://github.com/LWanzi/GANCL/blob/origin/GANCL.png)

Introduction
-----

We propose a novel self-supervised deep learning model called scGANCL for scRNA-seq data imputation. scGANCL combines generative adversarial network (GAN) with contrastive learning (CL) to improve imputation performance. scGANCL learns discriminative representation by distinguishing real samples from generated samples and utilizes the zero-inflated negative binomial (ZINB) distribution to model the original probability distribution of scRNA-seq data within the GAN framework. We evaluate scGANCL on various downstream analysis tasks, including gene expression recovery, clustering, trajectory inference, and differentially expressed genes (DEGs) analysis. The results showed that scGANCL outperformed four state-of-the-art methods consistently. Besides, ablation studies demonstrated the contributions of each component of scGANCL to the overall performance.

Requirement
-----
Python == 3.6.13

Pytorch==1.10.0

h5py==3.1.0

scanpy==1.7.2

umap-learn == 0.5.3

Usage
-----
You can run the scGANCL from the command line:

$ python main.py --dataset adam --epochs 200

Arguments
-----

|    Parameter    | Introduction                                                 |
| :-------------: | ------------------------------------------------------------ |
|    dataset     | A h5 file. Contains a matrix of scRNA-seq expression values,true labels, and other information. By default, genes are assumed to be represented by rows and samples are assumed to be represented by columns.|
|    task     |Downstream task, default is clustering |
|     epochs     | Number of training epochs                                    |
