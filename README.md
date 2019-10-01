# Repository for the paper "MEED: An Unsupervised Multi-Environment EventDetector for Non-Intrusive Load Monitoring" by Daniel Jorde and Hans-Arno Jacobsen

## Content
The repository contains the implementations of the four algorithms used for the benchmark of the MEED event detector and the MEED event detector itself.
The algorithms are implemented following the sklearn API.
The MEED event dector class contains the autoencoder model definition in its fit() function, but the training has do be done externally as it
is to computationally intensive and dependent on the dataset to be done within the fit() function.

### For questions plase contact: daniel.jorde@tum.de


