## Overview
In this work, we present the first systematic study on the role of different neurons in memorizing different group information, and confirm the existence of critical neurons where memorization of spurious correlations occurs.

We show that modifications to specific critical neurons can significantly affect model performance on the minority groups, while having almost negligible impact on the majority groups.

We propose spurious memorization as a new perspective on explaining the behavior of critical neurons in causing imbalanced group performance between majority and minority groups.

## Environment Preparation
The required packages are listed in `requirements.txt`. Make sure you having `.../SilentMajority` as the working directory when running any scripts.

## Dataset Preparation

### Waterbirds
The Waterbirds dataset is constructed by cropping out birds from photos in the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) and transferring them onto backgrounds from the Places dataset (Zhou et al., 2017).

Our code expects you set `.../waterbird_complete95_forest2water2` to the argument `--waterbird_dataset_path`.

You can download a tarball of this dataset [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz). 

### CelebA
Our code expects the `img_align_celeba` in the `celeba` directory:
```
.../celeba/img_align_celeba/
```
You need to set `.../celeba` to the argument `--celeba_dataset_path`.

You can download these dataset files from [here](https://www.kaggle.com/jessicali9530/celeba-dataset).

## Model Checkpoints Preparation
Download the following checkpoints to `experiments/` folder, keep the name unchanged:
| Model      | Pretrained Checkpoint Download Link                                                                                        |
|------------|----------------------------------------------------------------------------------------------------------------------------|
| resnet50   | https://download.pytorch.org/models/resnet50-19c8e357.pth                                                                  |
| deit_tiny  | https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth                                                     |
| deit_small | https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth                                                    |
| deit_base  | https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth                                                     |
| deit_large | https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth |

## Part I: Identifying the Existence of Critical Neurons
In this section, we validate the existence of critical neurons in the presence of spurious correlations.

| Bash Scripts                                         | Objectives                                                                                          |
|------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| runSpuriousCorrelationPretrainingErm.sh              | Train the model on celeba/waterbirds via ERM.                                                       |
| runStudyChannelwiseLargestNeuronExperiments.sh       | Investigate the impact of modifying the top-k largest neurons in a model trained using ERM.         |
| runStudyChannelwiseMostActivatedNeuronExperiments.sh | Investigate the impact of modifying the top-k most activated neurons in a model trained using ERM.  |
| runStudyLargestNeuronByThresholdExperiments.sh       | Investigate the impact of modifying the top-x percent largest neurons in a model trained using ERM. |

## Part II: Spurious Memorization by Critical Neurons
In this section, we take a further step in demystifying the cause of imbalanced group performance under spurious correlation, particularly focusing on the discrepancy in the test accuracy between majority and minority groups. 

The script `runSpuriousCorrelationFinetuningPruning.sh` aims to finetune the model with critical neuron pruned, in order to closing the gap between the majority and minority group accuracy.

