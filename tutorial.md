CSAW-HackML-2020 -- report
======

# Environment Initialization
```
    In order to separate user's existing environment with our code's required environment, we suggest to use conda to manage the environment, below are the steps to install conda & python environment:
    1. install conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
    2. create conda environment
        conda create --name [env_name] python=3.6
    3. activate env
        conda activate [env_name]
    
    To make the environment less ambigious, we write a requirements.txt for env preparation more easily, just use command would help you to install required python modules:
        pip install -r requirements.txt
```

# How to Test The Validation Data Accuracy

## How to train a repaired model from a badnet
- We accept two parameters for all model files "xx_model.py", command like this:
```
First parameter is used to load retrained validation data, second parameter is used to load poisoned model:
python baseline_model.py [clean_validation_data_filename] [poisoned_net_filename] [output_model_filename]
eg: python fine_pruning_model.py data/clean_test_data.h5 models/sunglasses_bd_net.h5 data/fine_pruning_model_G1.h5
```
- These files would produce an output model to file `output_model_filename` which could be used to test the performance

## How to test the performance of a repaired model
- Since different models have different architecture, the command could be different:
### random_pruning and fine_pruning model
```
python [random_pruning_eval.py|fine_pruning_eval.py] [test_data_filename] [poisoned_model_filename] [repaired_model_filename]
eg: python random_pruning_eval.py data/clean_test_data.h5 models/sunglass_bd_net.h5 fixed_models/repaired_random_pruning_model_G1.h5
```



