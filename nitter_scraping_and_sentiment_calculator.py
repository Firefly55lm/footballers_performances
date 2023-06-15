import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from datetime import timedelta
import pandas as pd
import re
from tqdm import tqdm
from csv_management import write_sentiment_csv
from convertions import convert_tweet_date
from textblob import TextBlob

headers = {"User-Agent" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, Like Gecko) "
                          "Chrome/47.0.2526.106 Safari/537.36 "}
page = "https://nitter.net/search?f=tweets&q="

analyzer = SentimentIntensityAnalyzer()

def tb_sentiment(tweet):
    """
        Converts polarity in emotion (TextBlob)
    """
    t = TextBlob(tweet)
    if t.polarity>0:
        return 'positive', t.polarity
    elif t.polarity == 0:
        return "neutral", t.polarity
    else:
        return "negative", t.polarity

def vad_sentiment(tweet):
    """
        Converts polarity in emotion (Vader)
    """
    if analyzer.polarity_scores(tweet)["compound"]>0:
        return "positive"
    elif analyzer.polarity_scores(tweet)["compound"]==0:
        return "neutral"
    else:
        return "negative"


def fix_query(query):
    return query.replace(" ", "+")

def set_page(url, name, since, until):
    """
        Returns Nitter's url
    """
    return url.split("q=")[0] + f"q={name}" + f"+lang:en&since={since}&until={until}"

def get_cursor(soup):
    """
        Returns Nitter's cursor
    """
    link_next_page = soup.find_all("div", {"class": "show-more"})
    try:
        return str(link_next_page).split('cursor=')[1].split('">Load more</a></div>')[0]
    except:
        return "none"

def clean_text(content):
    '''
        Regular expression that removes links and special characters from tweet
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(https?\S+)", " ", content).split())

def get_tweets_details(tweets, output, count, name, gamedate, since, direction):
    """
        Gets tweets' details from an HTML code
    """
    for tweet in tweets:
        if len(output) < count:
            content = re.findall('class="tweet-content media-body" dir="auto">' + "(.*?)" + '</div>', str(tweet))
            if content:
                tweetdate = "NA"
                content = clean_text(content[0])
                pol_vad = analyzer.polarity_scores(content)["compound"]
                sent_vad = vad_sentiment(content)
                pol_tb = tb_sentiment(content)[1]
                sent_tb = tb_sentiment(content)[0]

                stats = re.findall('class="tweet-date">' + "(.*?)" + '</a>', str(tweet))
                date = re.findall('title="' + "(.*?)" + ' · ', str(stats))

                try:
                    tweetdate = convert_tweet_date(date)
                except:
                    tweetdate = "NA"

                output.append((name, content, pol_vad, sent_vad, pol_tb, sent_tb, gamedate, tweetdate, direction))

def get_nitter_tweet_sentiment(query, count, gamedate, delta, direction):
    """
        The main function: gets tweets and calculates their sentiment
    """
    name = query
    gamedate_format = datetime.strptime(gamedate, "%Y-%m-%d")
    if direction == "before":
        since = (gamedate_format - timedelta(days=delta)).strftime("%Y-%m-%d")
        until = gamedate
    elif direction == "after":
        since = (gamedate_format + timedelta(days=1)).strftime("%Y-%m-%d")
        until = (gamedate_format + timedelta(days=delta+1)).strftime("%Y-%m-%d")

    collected_tweets = []
    query = fix_query(query)

    core = "https://nitter.net/search?f=tweets&q="
    page = set_page(core, query, since, until)

    flag = True
    while len(collected_tweets) < count and flag is True:
        old_len = len(collected_tweets)

        pageTree = requests.get(page, headers=headers)
        pageSoup = BeautifulSoup(pageTree.content, "html.parser")
        cursor = get_cursor(pageSoup)
        tweets = pageSoup.find_all("div", {"class": "timeline-item"})
        get_tweets_details(tweets, collected_tweets, count, name, gamedate, since, direction)
        page = page + '&cursor=' + cursor

        if cursor == "none":
            flag = False

        if len(collected_tweets) == old_len:
            flag = False

    return tuple(collected_tweets)


def get_sentiment_from_csv(csv_file):
    """
        Calls get_nitter_tweet_sentiment function for every player in the given dataset
    """
    premier = pd.read_csv(csv_file, encoding = "windows-1252")
    # premier_test = premier[:50]
    #print(premier.head())
    occurrences = tuple(premier.to_records())
    # occurrences = tuple(premier_test.to_records())
    #print(occurrences)

    result = []
    for occurrence in tqdm(occurrences):
        query = occurrence[4]
        gamedate = occurrence[8]

        for tweet in get_nitter_tweet_sentiment(query, 20, gamedate, 2, "before"):
            result.append(tweet)
        for tweet in get_nitter_tweet_sentiment(query, 20, gamedate, 2, "after"):
            result.append(tweet)

    return result

if __name__ == "__main__":
    to_write_sentiment = get_sentiment_from_csv("premier-league.csv")
    write_sentiment_csv(to_write_sentiment)
