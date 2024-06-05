# Welcome to nnUNet for pathology!
This repository contains code for applying nnUNet to pathology applications and is related to our [MIDL long paper](https://openreview.net/forum?id=aHuwlUu_QR). The [nnunet_for_pathology_v1 branch](https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v1) contains the code used in the paper, and guides you through the steps taken to achieve our #1 segmentation performance on the [TIGER challenge](https://tiger.grand-challenge.org/)'s experimental test set.

# nnUNet with dynamic dataloading!
The code available in **this branch** is built on nnUNet version 2, and comes with adjustable dataloaders that broaden the applicability of nnUNet to pathology data. Given the large size and high resolution of pathology Whole Slide Images (WSI), the static dataset approach of nnUNet may not meet the needs of all pathology segmentation tasks. hence, this version of nnUNet-for-pathology is equipped with dynamic dataloaders, that allows to sample from WSIs and corresponding Whole Slide Annotations: xml, json, or masks (that you can convert to xml or json using code that comes with this repo). 

# Main changes are:
1) A simplified workflow and train script
2) A new experiment planner
3) New configurable pathology dataloaders
4) Utility code with a WSI inference pipeline, code to convert label masks to xml or jsons, and to test your dataloading 

## A simplified workflow and train script ([click here](https://github.com/DIAGNijmegen/nnUNet-for-pathology/blob/nnunet_for_pathology_v2/pathology_code_and_utils/installs_and_run_training.sh))
With the use of dynamic dataloaders we do not need to preprocess and fingerprint a full dataset of static images and labels. Therefore all required files for training can directly be put in your `nnUNet_preprocessed folder`. This folder, as with usual nnUNet, should contain folders for dedicated datasets (for example Dataset001_TIGER_challenge). This dataset folder should contain 2 simple jsons: a `files.json` (containing the paths to your files), and a `dataset.json` (containing the mapping of the labels in your annotations, and optional label sampling weights). Next you should take a trainer name from the pathology trainers that suits your needs. Additionally you should add your `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results paths` (and optionally your `weights and biases API key`) to the simplified train script: [installs_and_run_training.sh](https://github.com/DIAGNijmegen/nnUNet-for-pathology/blob/nnunet_for_pathology_v2/pathology_code_and_utils/installs_and_run_training.sh).

Altogether you can train your model with the following command:

`bash installs_and_run_training.sh <dataset index> <fold> <trainer name>`

example:

`bash installs_and_run_training.sh 1 0 nnUNetTrainer_WSD_bal_nnunet_aug`

The script will:
1) Install your nnUNet folder as the nnunetv2 module
2) Export your nnUNet paths and wandb api key
3) Run the new pathology experiment planner using your `dataset.json` (and optionally your `GPU size`)
4) Train your specified `fold` on your files in the `files.json` (or `splits.json` if its present) using the specified `trainer`

In step 4 the model will search for a `splits.json` in the dataset folder of your nnUNet_preprocesed path. If it doesnt exits, it will take your `files.json` and split it randomly into 5 training/validation folds. Examples of the json files can be found [here](https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v2/pathology_code_and_utils/example_jsons).

# A new experiment planner ([click here](https://github.com/DIAGNijmegen/nnUNet-for-pathology/blob/nnunet_for_pathology_v2/nnunetv2/experiment_planning/experiment_planners/pathology_experiment_planner.py))
With the use of dynamic dataloading we do not need to fingerprint our full static dataset to find all preferred hyperparameters for your nnUNet model. Therefore, the experiment planning part could be simplified a lot, and some hyperparameters like the spacing should be defined in the Trainer. This may seem to bypass the idea of nnUNet to define all hyperparameters for you, but to make nnUNet v1 applicable on pathology, we needed to create a static dataset with a predefined spacing and arbitrary patch sizes, which influenced the fingerprint as well. The current experiment planner only needs your `dataset.json`, and optimizes hyperparameters like the batch and patch size, and the depth of the model, etc. that will be used by your configurable dataloaders. Additionally it will replace instance normalization with batch normalization in your model, as proposed in our [MIDL paper](https://openreview.net/forum?id=aHuwlUu_QR)

# New configurable pathology dataloaders ([click here](https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v2/nnunetv2/training/nnUNetTrainer/variants/pathology))
We included a set of pathology trainers that allow you to sample dynamically (on the fly) from WSIs and their annotations. The dataloaders work with the Plans generated by the new experiment planner. Trainers are configured using predefined templates along with a few settings (such as your label mapping from your `dataset.json`, and sampling strategy) that you need to specify yourself. 

All trainers currently inherit from the [nnUNetTrainer_WSD_undefined_dataloader](https://github.com/DIAGNijmegen/nnUNet-for-pathology/blob/nnunet_for_pathology_v2/nnunetv2/training/nnUNetTrainer/variants/pathology/nnUNetTrainer_WSD_undefined_dataloader.py) which is an adjusted default [nnUNetTrainer](https://github.com/DIAGNijmegen/nnUNet-for-pathology/blob/nnunet_for_pathology_v2/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py), and contains the default settings for pathology dataloading. This undefined_dataloader in itself is not functional, since it misses certain settings (hence its undefined) that are filled in by its child classes. It removed a lot of code that are no longer needed with our dynamic dataloading, and allows the construction of configurable dataloaders. In the current state we use the [WholeSlideData](https://github.com/DIAGNijmegen/pathology-whole-slide-data) library to construct configurable dataloaders, however, you are free to use this repository and put your own dataloader logic in the `get_dataloaders` function, that should simply return a train iterator and a val iterator on which the train loop can call `next()`.

### Trainer names
The currently available trainers have a structured name: 
- nnUNetTrainer_WSD_ +
- \<sampling strategy\> +
- \<ignore label 0\> +
- \<nnUNet or Albumentations augmentations\> +
- \<use xml or json annotations\>

### Sampling strategy
Current available sampling stategies are:
- `bal` (balanced) which will sample all lables (from your label mapping, specified in a dataset.json (see Simplified file structure section)) in a uniform manner
- `wei` (weighted) which will sample labels according to label sample weights, that you need to specify next to your label mapping
- `roi` which samples Regions Of Interest (ROIs), which is similar to default nnUNet dataloading, where individual static images (ROIs) are sampled.

### Ignore label 0
If regions of a WSI are not annotated, these pixels will get label 0, and are therefore 'unannoated' and not 'background' which is default nnUNet behaviour. Therefore trainers with `_i0` will make sure label 0 is treated as unannotated and will be ignored during loss calculation. Make sure you map 'true background' to 1 instead of 0 if you use `_i0` trainers. 

### Augmentation
We currently have 2 options for augmentation: nnUNet and Albumentations. nnunet augmention executes default nnnuent augmentations together with HED augmentation as proposed in our [MIDL paper](https://openreview.net/forum?id=aHuwlUu_QR). The implementation is somewhat different from our [v1 branch](https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v1), due to ongoing speed optimisation efforts. The other option is augmentation using Albumentations, where you can specify your complete set of augmentations in the trainer's template (which resides in the same folder).

### XML vs JSON annotations
XMLs are nice because they can be easily visualised in pathology image viewers like [ASAP](https://github.com/computationalpathologygroup/ASAP), but are slower and more working memory intensive during dataloading. 

# Utility code ([click here](https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v2/pathology_code_and_utils))
We added some examples, a WSI inference pipeline, code to convert masks to xml or json, and a notebook to test your training with its dataloader to [this](https://github.com/DIAGNijmegen/nnUNet-for-pathology/tree/nnunet_for_pathology_v2/pathology_code_and_utils) folder.

# Docker
We added a docker that we use in our group to the repo as well. If you encounter issues with dependencies, you might find a solution there.. 

# Please note that the current state of the repo is not polished. With the release of this code we intend to support the community and stimulate the use of nnUNet in our domain. 
---------------------------------------------------



# Original nnUNet (v2) readme follows:

# Welcome to the new nnU-Net!

Click [here](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) if you were looking for the old one instead.

Coming from V1? Check out the [TLDR Migration Guide](documentation/tldr_migration_guide_from_v1.md). Reading the rest of the documentation is still strongly recommended ;-)

## **2024-04-18 UPDATE: New residual encoder UNet presets available!**
Residual encoder UNet presets substantially improve segmentation performance.
They ship for a variety of GPU memory targets. It's all awesome stuff, promised! 
Read more :point_right: [here](documentation/resenc_presets.md) :point_left:

Also check out our [new paper](https://arxiv.org/pdf/2404.09556.pdf) on systematically benchmarking recent developments in medical image segmentation. You might be surprised!

# What is nnU-Net?
Image datasets are enormously diverse: image dimensionality (2D, 3D), modalities/input channels (RGB image, CT, MRI, microscopy, ...), 
image sizes, voxel sizes, class ratio, target structure properties and more change substantially between datasets. 
Traditionally, given a new problem, a tailored solution needs to be manually designed and optimized  - a process that 
is prone to errors, not scalable and where success is overwhelmingly determined by the skill of the experimenter. Even 
for experts, this process is anything but simple: there are not only many design choices and data properties that need to 
be considered, but they are also tightly interconnected, rendering reliable manual pipeline optimization all but impossible! 

![nnU-Net overview](documentation/assets/nnU-Net_overview.png)

**nnU-Net is a semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided 
training cases and automatically configure a matching U-Net-based segmentation pipeline. No expertise required on your 
end! You can simply train the models and use them for your application**.

Upon release, nnU-Net was evaluated on 23 datasets belonging to competitions from the biomedical domain. Despite competing 
with handcrafted solutions for each respective dataset, nnU-Net's fully automated pipeline scored several first places on 
open leaderboards! Since then nnU-Net has stood the test of time: it continues to be used as a baseline and method 
development framework ([9 out of 10 challenge winners at MICCAI 2020](https://arxiv.org/abs/2101.00232) and 5 out of 7 
in MICCAI 2021 built their methods on top of nnU-Net, 
 [we won AMOS2022 with nnU-Net](https://amos22.grand-challenge.org/final-ranking/))!

Please cite the [following paper](https://www.google.com/url?q=https://www.nature.com/articles/s41592-020-01008-z&sa=D&source=docs&ust=1677235958581755&usg=AOvVaw3dWL0SrITLhCJUBiNIHCQO) when using nnU-Net:

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.


## What can nnU-Net do for you?
If you are a **domain scientist** (biologist, radiologist, ...) looking to analyze your own images, nnU-Net provides 
an out-of-the-box solution that is all but guaranteed to provide excellent results on your individual dataset. Simply 
convert your dataset into the nnU-Net format and enjoy the power of AI - no expertise required!

If you are an **AI researcher** developing segmentation methods, nnU-Net:
- offers a fantastic out-of-the-box applicable baseline algorithm to compete against
- can act as a method development framework to test your contribution on a large number of datasets without having to 
tune individual pipelines (for example evaluating a new loss function)
- provides a strong starting point for further dataset-specific optimizations. This is particularly used when competing 
in segmentation challenges
- provides a new perspective on the design of segmentation methods: maybe you can find better connections between 
dataset properties and best-fitting segmentation pipelines?

## What is the scope of nnU-Net?
nnU-Net is built for semantic segmentation. It can handle 2D and 3D images with arbitrary 
input modalities/channels. It can understand voxel spacings, anisotropies and is robust even when classes are highly
imbalanced.

nnU-Net relies on supervised learning, which means that you need to provide training cases for your application. The number of 
required training cases varies heavily depending on the complexity of the segmentation problem. No 
one-fits-all number can be provided here! nnU-Net does not require more training cases than other solutions - maybe 
even less due to our extensive use of data augmentation. 

nnU-Net expects to be able to process entire images at once during preprocessing and postprocessing, so it cannot 
handle enormous images. As a reference: we tested images from 40x40x40 pixels all the way up to 1500x1500x1500 in 3D 
and 40x40 up to ~30000x30000 in 2D! If your RAM allows it, larger is always possible.

## How does nnU-Net work?
Given a new dataset, nnU-Net will systematically analyze the provided training cases and create a 'dataset fingerprint'. 
nnU-Net then creates several U-Net configurations for each dataset: 
- `2d`: a 2D U-Net (for 2D and 3D datasets)
- `3d_fullres`: a 3D U-Net that operates on a high image resolution (for 3D datasets only)
- `3d_lowres` → `3d_cascade_fullres`: a 3D U-Net cascade where first a 3D U-Net operates on low resolution images and 
then a second high-resolution 3D U-Net refined the predictions of the former (for 3D datasets with large image sizes only)

**Note that not all U-Net configurations are created for all datasets. In datasets with small image sizes, the 
U-Net cascade (and with it the 3d_lowres configuration) is omitted because the patch size of the full 
resolution U-Net already covers a large part of the input images.**

nnU-Net configures its segmentation pipelines based on a three-step recipe:
- **Fixed parameters** are not adapted. During development of nnU-Net we identified a robust configuration (that is, certain architecture and training properties) that can 
simply be used all the time. This includes, for example, nnU-Net's loss function, (most of the) data augmentation strategy and learning rate.
- **Rule-based parameters** use the dataset fingerprint to adapt certain segmentation pipeline properties by following 
hard-coded heuristic rules. For example, the network topology (pooling behavior and depth of the network architecture) 
are adapted to the patch size; the patch size, network topology and batch size are optimized jointly given some GPU 
memory constraint. 
- **Empirical parameters** are essentially trial-and-error. For example the selection of the best U-net configuration 
for the given dataset (2D, 3D full resolution, 3D low resolution, 3D cascade) and the optimization of the postprocessing strategy.

## How to get started?
Read these:
- [Installation instructions](documentation/installation_instructions.md)
- [Dataset conversion](documentation/dataset_format.md)
- [Usage instructions](documentation/how_to_use_nnunet.md)

Additional information:
- [Learning from sparse annotations (scribbles, slices)](documentation/ignore_label.md)
- [Region-based training](documentation/region_based_training.md)
- [Manual data splits](documentation/manual_data_splits.md)
- [Pretraining and finetuning](documentation/pretraining_and_finetuning.md)
- [Intensity Normalization in nnU-Net](documentation/explanation_normalization.md)
- [Manually editing nnU-Net configurations](documentation/explanation_plans_files.md)
- [Extending nnU-Net](documentation/extending_nnunet.md)
- [What is different in V2?](documentation/changelog.md)

Competitions:
- [AutoPET II](documentation/competitions/AutoPETII.md)

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)

## Where does nnU-Net perform well and where does it not perform?
nnU-Net excels in segmentation problems that need to be solved by training from scratch, 
for example: research applications that feature non-standard image modalities and input channels,
challenge datasets from the biomedical domain, majority of 3D segmentation problems, etc . We have yet to find a 
dataset for which nnU-Net's working principle fails!

Note: On standard segmentation 
problems, such as 2D RGB images in ADE20k and Cityscapes, fine-tuning a foundation model (that was pretrained on a large corpus of 
similar images, e.g. Imagenet 22k, JFT-300M) will provide better performance than nnU-Net! That is simply because these 
models allow much better initialization. Foundation models are not supported by nnU-Net as 
they 1) are not useful for segmentation problems that deviate from the standard setting (see above mentioned 
datasets), 2) would typically only support 2D architectures and 3) conflict with our core design principle of carefully adapting 
the network topology for each dataset (if the topology is changed one can no longer transfer pretrained weights!) 

## What happened to the old nnU-Net?
The core of the old nnU-Net was hacked together in a short time period while participating in the Medical Segmentation 
Decathlon challenge in 2018. Consequently, code structure and quality were not the best. Many features 
were added later on and didn't quite fit into the nnU-Net design principles. Overall quite messy, really. And annoying to work with.

nnU-Net V2 is a complete overhaul. The "delete everything and start again" kind. So everything is better 
(in the author's opinion haha). While the segmentation performance [remains the same](https://docs.google.com/spreadsheets/d/13gqjIKEMPFPyMMMwA1EML57IyoBjfC3-QCTn4zRN_Mg/edit?usp=sharing), a lot of cool stuff has been added. 
It is now also much easier to use it as a development framework and to manually fine-tune its configuration to new 
datasets. A big driver for the reimplementation was also the emergence of [Helmholtz Imaging](http://helmholtz-imaging.de), 
prompting us to extend nnU-Net to more image formats and domains. Take a look [here](documentation/changelog.md) for some highlights.

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
