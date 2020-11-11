# Evaluation of SCGAN

## 1 Methods

We proposed three kinds of evaluators, and named them as follows:

- **Traditional indexes (Reference based)**: PSNR, SSIM, MSE, and NRMSE;

- **New indexes (No reference)**: CCI, CCI Ratio (the ratio of the number of generated images in optimum range to the whole validation images), CNI;

- **VGG16 indexes (Classification task based)**: Top-1 Accuracy and Top-5 Accuracy on validation dataset of ImageNet.

For more information about CCI and CNI, please see the original [paper](https://www.sciencedirect.com/science/article/pii/S1077314206000233).

For VGG16 indexes, please enter the folder `global feature network pre train`.

## 2 Usage

Please run `validation.py` to get results of traditional indexes and new indexes.

Please run `validation.py` and change `baseroot` parameter to get results of VGG16 indexes.
