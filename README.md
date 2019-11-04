# Specialization Project

MS specialization project for Cornell-Technion Dual degree program in Information Systems by Noel Konagai.

## Setup

We use a Python 3.7 virtual environment in Conda. To set this up:

```bash
conda create -n <your_env> python=3.7 anaconda
```

To activate:

```bash
source activate <your_env>
```

Install the requirements from the repo:

```bash
pip install -r requirements.txt
```

To deactivate:

```bash
deactivate
```

## Testing

``test.py`` lets you pass in custom arguments to test certain parts of ``graph.py``.

|flag|desc|required?|type|choices|used in|
|-|-|-|-|-|-|
|-f|which function to run|yes|str|vary_char, vary_time|N/A|
|-m|which regression model to run|yes|str|all, lasso, decision_tree, ridge, svm, voting|N/A|
|-t|what time difference to use (mins)|no|multiple int|vary_time|N/A|
|-l|what minimum character length to use|no|multiple int|N/A|vary_char|
|-s|saving the output to numpy for further visualization|no|bool|N/A|both|

So, for example if you want to run a voting regression on a dataframe where conversations were segmented by various message character lengths:

```bash
python test.py -f vary_char -m voting -l 70 75 80 85 90
```

