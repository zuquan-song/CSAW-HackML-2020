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
- We accept three parameters for all model files "xx_model.py", command like this:
```
First parameter is used to load retrained validation data, second parameter is used to load poisoned model:
python fine_pruning_model.py [clean_validation_data_filename] [poisoned_net_filename] [output_model_filename]
eg: 
python random_pruning_model.py data/clean_validation_data.h5 models/anonymous_1_bd_net.h5 data/random_pruning_model_for_anonymous_1_bd_net.h5
python random_pruning_model.py data/clean_validation_data.h5 models/anonymous_2_bd_net.h5 data/random_pruning_model_for_anonymous_2_bd_net.h5
python random_pruning_model.py data/clean_validation_data.h5 models/multi_trigger_multi_target_bd_net.h5 data/random_pruning_model_for_multi_trigger_bd_net.h5
python random_pruning_model.py data/clean_validation_data.h5 models/sunglasses_bd_net.h5 data/random_pruning_model_for_sunglasses_bd_net.h5

python fine_pruning_model.py data/clean_validation_data.h5 models/anonymous_1_bd_net.h5 data/fine_pruning_model_for_anonymous_1_bd_net.h5
python fine_pruning_model.py data/clean_validation_data.h5 models/anonymous_2_bd_net.h5 data/fine_pruning_model_for_anonymous_2_bd_net.h5
python fine_pruning_model.py data/clean_validation_data.h5 models/multi_trigger_multi_target_bd_net.h5 data/fine_pruning_model_for_multi_trigger_bd_net.h5
python fine_pruning_model.py data/clean_validation_data.h5 models/sunglasses_bd_net.h5 data/fine_pruning_model_for_sunglasses_bd_net.h5
```
- These files would produce an output model to file `output_model_filename` which could be used to test the performance

## How to test the performance of a repaired model
- Since different models have different architecture, the command could be different:
### rule based model
```
python rule_based_model_eval.py [test_data_filename] [poisoned_model_filename] [repaired_model_filename]
eg: python rule_based_model_eval.py data/clean_test_data.h5 models/sunglasses_bd_net.h5 fixed_models/random_pruning_model_for_anonymous_1_bd_net.h5
```
### autoencoder based model
```
python autoencoder_based_model_eval.py [test_data_filename] [repaired_model_filename]

eg: python autoencoder_based_model_eval.py data/clean_test_data.h5 fixed_models/random_pruning_model_for_anonymous_1_bd_net.h5
```

## How to Run the report result
```
python benchmark.py
```


