# Data Cleaning

Document created by Noel Konagai.

## Summary

The ```csv_to_pandas.ipynb``` notebook takes the input of an exported Whatsapp conversation and converts it into a Pandas dataframe that is easier to analyze. Through this process, the messages and other attributes of the row data are cleaned. Last, the notebook translates the messages to a common language, English, to ease the process of analysis.

## Dependencies

Install requirements using the command line.

```python
pip install -r requirements.txt
```

## Functions Used

Date conversion into a machine readable date: ```convert_date(date_time)```

Creating a list of emojis from a given message: ```split_count(text)```

Stripping the given message of emojis for ease of analysis: ```demojize(text)```

Creating a list of URLs from a given message: ```find_url(text)```

## Translation

Soon to be added.
