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

# How to Test Your Validation Data Accuracy
```
   To test the accuracy of baseline_model, use command:
    python baseline_model.py [your_test_data_filename] [your_net_filename] 
   
   eg: python baseline_model.py data/clean_test_data.h5 models/sunglasses_bd_net.h5 
```




