CSAW Project Report
====
- Project Members: Chengyu Jiang (cj1573), Yunzhao Xu (yx2086), Zuquan Song (zs1243)
- Contribution: Chenyu Jiang (33.3%), Yunzhao Xu (33.3%), Zuquan Song(33.3%)

# Code
## Structure
```
.
├── README.md
├── architecture.py
├── baseline_model.py # contains baseline_model information
├── benchmark.py    # contains benchmark for different repaired models
├── data
│   ├── clean_test_data.h5
│   ├── clean_validation_data.h5
│   ├── data.txt
│   └── sunglasses_poisoned_data.h5
├── eval.py     # this file accept 
├── jupyter # jupyter folder contains intermediate progress of our project, which could be ignored
│   ├── Final.ipynb
│   ├── experimental.ipynb
├── models
│   ├── anonymous_bd_net.h5
│   ├── anonymous_bd_weights.h5
│   ├── repaired_model_baseline.h5
│   ├── sunglasses_bd_net.h5
│   └── sunglasses_bd_weights.h5
├── report.md
├── requirements.txt
└── utils.py # contains model evaluation class
```
### Baseline Repaired Model
#### Pipeline

#### Source
 - Liu, Kang, Brendan Dolan-Gavitt, and Siddharth Garg. "Fine-pruning: Defending against backdooring attacks on deep neural networks." International Symposium on Research in Attacks, Intrusions, and Defenses. Springer, Cham, 2018.

#### Performance
