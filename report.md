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
#### Assumption
We may assume that the attacker tried to use the pruning aware attack strategy.
In this strategy the attacker implement the four step:
1. Trains the baseline DNN on a clean training data set;
2. Prunes the DNN by eliminating dormant neurons;
3. Re-trains the pruned DNN, but this time with the poisoned training dataset.
4. Re-instating all pruned neurons back into the network along with the associated weights and biases

In our case, the attacker implemented a targeted backdoor attack on face recognition where a specific pair of sunglasses in badnet B1.
And in bad net B2, B3, the attacker using the attack strategy discussed above.
#### Pipeline
According to the assumption, we performed a fine-pruning defense on B1, also in B2, B3.

Firstly, we use the weights from attacker and load it with bad net, this step is used for fine-tune.

Then we use validation data to train the badnet and iteratively prunes neurons from the DNN in increasing order of average activations and records the accuracy of the pruned network in each iteration. Remove the removes decoy neurons.

Finally, we use validation data to fine-tune the model and produce a repaired one.

To recognize the backdoor as an N + 1 class, we compare the predict result from original badnet and repaired one, if the output class is the same, it shows that the original data is not tainted by sunglasses, else it can be recognized as an N + 1 class. 
#### Result
After the fine-pruning step, the accuracy of the repaired model on test data is .
#### Source
 - Liu, Kang, Brendan Dolan-Gavitt, and Siddharth Garg. "Fine-pruning: Defending against backdooring attacks on deep neural networks." International Symposium on Research in Attacks, Intrusions, and Defenses. Springer, Cham, 2018.

#### Performance
