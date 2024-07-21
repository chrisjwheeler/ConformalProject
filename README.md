# Conformal Prediction Research Project
## About

This repository is a research project supervised by Dr. Henry Reeve focusing on conformal techniques for online learning under distribution shift.

## Structure
To aid the research I have create a python package `ConformalMethods` which is located in the `src` directory. This package contains three classes for generating data, running conformal techniques and plotting/comparing the results. There is an [examples notebook](src/Example.ipynb) which gives a brief indiication of the capabilites of the package. I am yet to write proper documentation.

The research I have done for the project is jupyter based and located in the `notebooks` directory. Each notebook follows the naming convetion XX_ ... which roughly indicates the order that each notebook makes sense in. Files containing 'legacy' indicate that they do not use the full capabilites of the `ConformalMethods` package as they predate it.

Finally in the `scripts` directory there is CLI for easily testing and comparing the different methods on financial data from yahoo finance.
