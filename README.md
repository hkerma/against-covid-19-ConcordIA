# Team ConcordIA - 4th place
Accuracy **>96%** on covidx-cxr2 test set. Score of **14.60/16** on competition test. 

![rank](https://gillesschneider.github.io/me/assets/images/rank.png)

# About our work
Go [here](https://gillesschneider.github.io/me/against-covid-19.html) for more details.

# Requirements

- python3.8
- [tqdm](https://pypi.org/project/tqdm/)
- [PIL](https://pypi.org/project/Pillow/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [sklearn](https://pypi.org/project/scikit-learn/)
- [numpy](https://pypi.org/project/numpy/)
- [torch](https://pypi.org/project/torch/)
- [torchvision](https://pypi.org/project/torchvision/)

# How to use

## For judging:
1. Get the competition dataset from [Kaggle](https://www.kaggle.com/andyczhao/covidx-cxr2)
2. Install the requirements (see list above)
3. Put the dataset in `<this_repo>/../dataset/competition_test` (see example tree below)
4. Run `python3 review.py`. The script will create a `submission_review.txt` file at the root of the repo. That file corresponds to our final submission on [eval.ai](https://eval.ai/web/challenges/challenge-page/925/leaderboard/2424)

Feel free to reach us for any question/inquiries.

Example tree:
```
├── against-covid-19-ConcordIA
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

## For training

Download the dataset from [here](https://www.kaggle.com/andyczhao/covidx-cxr2) and extract the files to `<CWD>/../dataset`.
Run `python3 main.py` to train the model, saved models are in `/saved_models`
