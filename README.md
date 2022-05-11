# SAM-kNN Regressor for Online Learning in Water Distribution Networks

This repository contains the implementation of the methods proposed in the paper [SAM-kNN Regressor for Online Learning in Water Distribution Networks](paper.pdf) by Jonathan Jakob, André Artelt, Martina Hasenjäger and Barbara Hammer.

The experiments as described in the paper are implemented in the folder [Implementation](Implementation/).

## Abstract
Water distribution networks are a key component of modern infrastructure for housing and industry. They transport and distribute water via widely branched networks from sources to consumers.
In order to guarantee a working network at all times, the water supply company continuously monitors the network and takes actions when necessary -- e.g. reacting to leakages, sensor faults and drops in water quality. Since real world networks are too large and complex to be monitored by a human, algorithmic monitoring systems have been developed. A popular type of such systems are residual based anomaly detection systems that can detect events such as leakages and sensor faults. For a continuous high quality monitoring, it is necessary for these systems to adapt to changed demands and presence of various anomalies.

In this work, we propose an adaption of the incremental SAM-kNN classifier for regression to build a residual based anomaly detection system for water distribution networks that is able to adapt to any kind of change.

## Details
### Implementaton of experiments
The shell script `run_experiments.sh` runs all experiments 

### Other (important) stuff
#### `SAM_KNN_Regression.py`
Implementation of our proposed SAM-kNN regressor.

## Data

Note that we did not publish the data sets due to large file sizes. Please contact us if you are interested in the data sets.

## Requirements

- Python3.6
- Packages as listed in `Implementation/REQUIREMENTS.txt`

## License

MIT license - See [LICENSE](LICENSE)
