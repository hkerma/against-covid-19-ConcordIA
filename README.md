# Team ConcordIA - 4th place
Accuracy **>96%** on covidx-cxr2 dataset. Score of **14.60/16**. 
![rank](https://gillesschneider.github.io/me/assets/images/rank.png)

# About our work
Go [here](https://gillesschneider.github.io/me/against-covid-19.html) for more details.

# How to use
```
├── against-covid-19-public
│   ├── dataset
│   │   ├── dataset.py
│   │   ├── test.csv
│   │   └── train.csv
│   ├── learning_curves
│   │   └── <learning_curves>.png
│   ├── main.py
│   ├── models
│   │   └── resnet.py
│   ├── README.md
│   ├── saved_models
│   │   ├── sub_4.pt
│   │   └── sub_5.pt
│   ├── submissions
│   │   ├── sub_1.txt
│   │   ├── sub_2.txt
│   │   ├── sub_3.txt
│   │   ├── sub_4.txt
│   │   ├── sub_5.txt
│   │   └── sub_6.txt
│   ├── toolsp
│   │   ├── equalization.py
│   │   ├── score.py
│   │   ├── train_test.py
│   │   └── visualization.py
│   │
│   │   ### You are here
│   └── tree.txt
│
│   ### Add here the downloaded dataset
└── dataset
    ├── competition_test
    │   └── <1-400>.png
    ├── test
    │   └── <test_images>.png/jpg
    ├── test.txt
    ├── train
    │   └── <train_images>.png/jpg
    └── train.txt
```

Download the dataset from [here](https://www.kaggle.com/andyczhao/covidx-cxr2) and extract the files to `<CWD>/../dataset`.
Run `main.py` to train the model, saved models are in `/saved_models`
