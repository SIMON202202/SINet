# SIMON
- This repository is to reproduce the results of an anonymous paper.
- A preprocessing generates **ten different splittings of train/validation/test sets** from simulation data. Then we run experiments with our proposed method and other baselines.

### Requirements
* python 3
* To install requirements:

```setup
conda env create -f environment.yml
conda activate simon_env
```

### Preprocessing 
* The simulation data can be download from [here](https://1drv.ms/u/s!AvkPhNiV_FS7ah_SCkYugU1Qc4g?e=HSQfZM) and should be set in the folder `./data/source/`.
* after unzip the file, run "bash ./preprocess/script_generatedata.sh"

### Main analysis
* see `./script./` for commands for running scripts.
* Further details are documented within the code.
