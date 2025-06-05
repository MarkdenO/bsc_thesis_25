
# BSc Thesis Information Science
## Automatic classification of Advent of Code solutions via code representations and machine learning

Welcome to the repository of my thesis for the BSc Information Science at the University of Groningen. 

The steps to reproduce my research are described below.

## Requirements
To reproduce this research, it is important to have the correct packages installed. 
```
$ python3.10 -m venv venv
$ pip3 install -r requirements.txt
```

## Data collection
The data for this thesis is collected from Reddit, GitHub (both non-leaderboard and leaderboard solutions). To collect the data execute the following steps:

Store your GitHub token in your environment to prevent rate limits being hit.
```
$ export GITHUB_TOKEN=your_token

$ python3 get_solutions.py --leaderboard --reddit --file 'path_to_additional_github_repos' 
```

## Preprocess and labelling
When all data is collected, we preprocess and label the data. This is done by the following steps:

Remove comments, docstrings, and imports:
```
$ python3 preprocess.py
```
To label the data, make sure that `categorization.csv` is in the same directory and execute:
```
$ python3 label_data.py
```

To preprocess the n-grams datasets:
```
$ python3 transform.py --ngrams
```

## Train and evaluate models
For the bag-of-words and n-grams models, use:
```
$ python3 train_models.py
```

For the AST model execute `ast_model.py`:
```
$ python3 ast_model.py
```

For the code embeddings model, run:
```
$ python3 contastive_learning.py
```