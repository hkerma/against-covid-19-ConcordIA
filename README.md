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
3. Run `python3 eval.py --list ["<path1>", "<path2>", ...] --model <path_to_model>`. The output corresponds to our final submission on [eval.ai](https://eval.ai/web/challenges/challenge-page/925/leaderboard/2424). By default, `path_to_model = ./saved_models/model.pt`.

Example: `python3 eval.py --list ["/absolute/path/to/1.png", "/absolute/path/to/2.png", ...] --model ./saved_models/model.pt`.

IMPORTANT NOTICE: note that due to the rules, you need to precisely respect the formatting (put a space between each image path, in an array, etc.).

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
