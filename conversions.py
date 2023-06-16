from datetime import datetime, timedelta
from operator import itemgetter
import string

months = {"jan" : "01", "feb" : "02", "mar" : "03", "apr" : "04",
          "may" : "05", "jun" : "06", "jul" : "07", "aug" : "08",
          "sep" : "09", "oct" : "10", "nov" : "11", "dec" : "12"}

def convert_date(date, months):
    """
        Converts the date from a month-day format (Es. Apr 05) to a year-month-day format (Es. 2023-04-05)
    """

    if date == "Yesterday":
        yesterday = datetime.now() - timedelta(days = 1)
        date = yesterday.strftime("%Y-%m-%d")
        return date
    elif date == "Today":
        today = datetime.now()
        date = today.strftime("%Y-%m-%d")
        return date
    else:
        if len(date.split(",")) > 1:
            year = str(date.split(",")[1].replace(" ", ""))
        else:
            year = "2023"

        month = str(date).split()[0].lower()
        day = str(date).split()[1]
        month = months[month]
        return str(year + "-" + month + "-" + day).replace(",", "")

def convert_tweet_date(date):
    """
        Converts tweet's date to a year-month-day format
    """
    date = str(date[0]).replace(",", "").split()
    month = date[0].lower()
    day = date[1]
    year = date[2]
    return year + "-" + months[month] + "-" + day

def emotion(pol):
    """
        Converts polarity in emotion
    """
    if pol == 0.0:
        return "neutral"
    elif pol > 0:
        return "positive"
    elif pol < 0:
        return "negative"
    else:
        return float('nan')

def char_rating(rating):
    """
        Converts a numeric rating into qualitative rating
    """
    try:
        rating = float(rating)
        if rating <= 5.5:
            return "very bad"
        elif rating < 6 and rating > 5.5:
            return "bad"
        elif rating >= 6.0 and rating < 7:
            return "sufficient"
        elif rating >= 7.0 and rating < 8:
            return "good"
        elif rating >= 8.0:
            return "excellent"
        else:
            return "NA"
    except:
        return "NA"

