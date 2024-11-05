# Genetic Programming for Continuous Facial Expression Recognition

This project explores Facial Expression Recognition (FER) in the continuous arousal-valence (AV) space using genetic programming and symbolic regression techniques. It aims to develop more efficient and explainable methods for capturing nuanced emotions in unconstrained environments.

## Features

* Feature learning using genetic programming with image descriptors (IDGP)
* Feature Learning with Symbolic Regression for interpretable model generation (FD-GPSR)
* Two-stage GP method for feature extraction and symbolic regression (CFSR)
* Hybrid model integrating EfficientNet-based feature extraction with GP-based symbolic regression
* Models comparison among the mentioned approaches and with baseline deep learning models (EfficientNet-based FER)

## Installation

1. Clone this repository
2. Install required dependencies
```
pip install numpy scikit-learn deap scoop h5py joblib opencv-python tensorflow pytorch pysr
```

## Usage
1. Preprocess the AffectNet dataset using the provided scripts
2. Run the desired model:
    * For IDGP: `python idgp_affectnet.py`
    * For FD-GPSR: `python idgp_srpysr_affectnet.py`
    * For CFSR: `python idgp_srdeap_affectnet.py`
    * For the hybrid model: `python nn_pysr`

## Data Sources
The AffectNet database provides raw images containing facial images annotated with categorical emotions and dimensional affect (valence and arousal) labels. A subset of images is sampled from the training and validation sets to create smaller, balanced datasets for each emotion category. The sampled images undergo several preprocessing steps:
    * Face alignment using a pre-trained face detection model
    * Grayscale conversion
    * Histogram equalization
    * Noise reduction
    * Resizing to 128x128 pixels

The preprocessed images are then combined with their corresponding arousal, valence, and expression labels. `train_images_data_yy_128px.h5 (400 images)` and `test_images_data_yy_128px.h5` (80 images) are two main files used in this project, particularly for IDGP, FD-GPSR, and CFSR models. `test_images_data_av80.h5` (80 images) and `train_images_data_av160.h5` (160 images) are even smaller subsets of the data. `test_features_yy_128px.h5` (80 images) and `train_features_yy_128px_3kimg.h5` (3840 images) are datasets used to train the hybrid model, which uses deep learning for feature extraction as input for symbolic regression prediction.

## Technologies Used
Python, NumPy, scikit-learn, DEAP (Distributed Evolutionary Algorithms in Python), SCOOP (Scalable COncurrent Operations in Python), OpenCV, Pytorch (for EfficientNet baseline)

## Screenshots

## License

The programs in this repository are provided for research purposes only

## Project Report

The project report covers the background, methodology, experiments, and results of using genetic programming and symbolic regression techniques for FER in the continuous AV space.

The report discusses the challenges of capturing nuanced emotions in unconstrained environments and highlights the need for more efficient and explainable methods. It presents several approaches investigated in this project, including feature learning using GP with image descriptors (IDGP), symbolic regression for interpretable model generation (FD-GPSR), a two-stage GP method (CFSR), and a hybrid model combining EfficientNet-based feature extractor with GP-based symbolic regression.

Experiments conducted on the AffectNet dataset demonstrate the effectiveness of the proposed methods, particularly the GPSR-based feature learning FD-GPSR approach. The report also analyzes the evolved features and equations to gain insights into the key facial regions and mathematical relationships captured by the models for AV prediction.

Key points for models comparison:
1. Feature Extraction:
* The EfficientNet model performs end-to-end feature learning.
* IDGP, FD-GPSR, and CFSR use GP to evolve feature extraction primitives.
* The hybrid model leverages EfficientNet for feature extraction.

2. Interpretability:
* The GP-based models (IDGP, FD-GPSR, CFSR) offer more interpretable solutions compared to the deep learning baseline.
* The symbolic regression approaches (FD-GPSR, CFSR, and hybrid model) provide explicit mathematical equations for AV prediction.

3. Adaptability:
* The GP-based models can automatically adapt their structure to the problem, potentially discovering novel feature combinations.
* The deep learning baseline relies on a fixed architecture, though it's highly optimized for image tasks.

For more details, please refer to the project report, project_report.pdf, included in this repository.
