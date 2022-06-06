# Building Robust Ensembles via Margin Boosting

Implementation of our ICML 2022 paper [Building Robust Ensembles via Margin Boosting](), 
by Dinghuai Zhang, Hongyang Zhang, Aaron Courville, Yoshua Bengio, Pradeep Ravikumar, and Arun Sai Suggala.

We propose a margin-boosting framework for learning
max-margin ensembles. 
Drawing theoretical insights from our boosting framework, we propose the MCE loss, 
a new variant of the CE loss, which improves the adversarial robustness of state-of-the-art defenses.

To get quick understanding of our proposed MCE loss, run

```
python main.py
python main.py --other_weight 1
```

For the robust boosting algorithms, run
```
python ladder_main.py --ensemble_num 5 --save 
python ladder_mce_main.py --ensemble_num 5 --save --other_weight 1
```