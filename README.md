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

``analyze.py`` lets you pass in custom arguments to test certain parts of ``chkpt3_analysis.py``.

|flag   |desc                                                |required? |type           |choices                |used in    |
|-      |-                                                   |-         |-              |-                      |-          |
|-f     |which function to run                               |no        |str            |vary_char, vary_time   |N/A        |
|-t     |what time difference to use (mins)                  |no        |multiple int   |N/A                    |vary_time  |
|-c     |what minimum character length to use                |no        |multiple int   |N/A                    |vary_char  |
|-r     |regression to run on all groups together or separate|no        |str            |all, separate          |N/A        |
|-d     |what data type to use                               |no        |str            |vary_char, vary_time   |N/A        |

So, for example if you want to dissect the dataframe with specific character cutoffs:

```bash
python analyze.py -f vary_char -c 60 70 80
```

These create individual CSV files either under ``data/chkpt3/conversation/time`` or ``~/char``. Once you have created the CSV files, you can run a regression to see how well the model predicts response rates. To do so:

```bash
python analyze.py -r separate -d vary_char
```