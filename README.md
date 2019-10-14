[![DOI](https://zenodo.org/badge/212113635.svg)](https://zenodo.org/badge/latestdoi/212113635)

# Repository for the paper "MEED: An Unsupervised Multi-Environment Event Detector for Non-Intrusive Load Monitoring"
### by Daniel Jorde and Hans-Arno Jacobsen

## Content
The repository contains the implementations of the four algorithms used for the benchmark of the MEED event detector and the MEED event detector itself.

All code is released under the MIT licence.

The algorithms are implemented following the sklearn API.
The hyperparameter settings used to produce the results in the paper are based on a grid search and are included as default values for the respective parameters in the algorithm implementations.

The MEED event dector class contains the autoencoder model definition in its fit() function, but the training has do be done externally as it is to computationally intensive and dependent on the dataset to be done within the fit() function.

Trained models can be found in the *MEED_Models* folder. They are stored as keras ".h5" models.

As the models are trained via cross-validation each subfolder contains multiple instances of each model, each having a unique id that corresponds to the fold of the cross-validation that was used to train them.

The *Notebooks* folder contains working examples in jupyter notebooks on one exemplary file of BLUED.
The notebooks show how the algorithms can be used and implemented in other NILM scenarios.
We highly suggest using these algorithms for benchmarks or other NILM papers.

Each of the algorithm classes also provides a *score* function that can compute the scores as we have done it in our paper.

This ensures other researchers can compare their algorithms with ours.

The score function uses a tolerance limit, like discussed in the paper and introduced by multiple other authors in the field,
to determine true positive events.

The *requirements.txt* file contains all python packages necessary.


### For questions please contact: daniel.jorde@tum.de


