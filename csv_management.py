import csv
import pandas as pd
import warnings
from tabulate import tabulate
from convertions import emotion, char_rating

def write_occurrences_csv(league_name, occurrences):
    """
    Writes the occurrences in a csv named league_name.csv
    """

    with open(f'{league_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["team_id", "team_name", "player_id", "player_name", "nationality",
             "age", "primary_pos", "date", "rating",
             "minutes", "goal", "assist"])
        for occurence in occurrences:
            writer.writerow(
                [occurence[0], occurence[1], occurence[2], occurence[3], occurence[4], occurence[5], occurence[6],
                 occurence[7], occurence[8], occurence[9], occurence[10], occurence[11]])
        file.close()

    print(f"\n\033[32mCOMPILED CSV FILE AVAILABLE: {league_name}.csv\033[0m")

def write_sentiment_csv(occurrences):
    """
    Writes the occurrences in a csv
    """

    with open('sentiment-dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["player_name", "text", "vader_polarity", "vader_emotion", "tb_polarity", "tb_emotion", "game_date", "tweet_date", "when"])
        for occurence in occurrences:
            writer.writerow(
                [occurence[0], occurence[1], occurence[2], occurence[3], occurence[4], 
                 occurence[5],occurence[6], occurence[7], occurence[8]])
        file.close()

    print(f"\n\033[32mCOMPILED CSV FILE AVAILABLE: sentiment-dataset.csv\033[0m")

def merge_premier_tweets():
    """
    Writes the dataframe in a csv named premier_with_sentiment.csv
    """

    warnings.filterwarnings("ignore")
    flag = False

    # OPENING THE TWO FILES
    try:
        premier = pd.read_csv("premier-league.csv", encoding="windows-1252")
        tweets = pd.read_csv("sentiment-dataset.csv", encoding="windows-1252")
        flag = True
    except:
        print(f"\n\033[31mMISSING REQUIRED FILES\033[0m")

    if flag is True:
        # SELECTION OF THE USEFUL FEATURES
        tweets = tweets[["player_name", "vader_polarity", "tb_polarity", "game_date", "when"]]
        premier = premier.rename(columns={"date": "game_date"})
        premier['char_rating'] = premier['rating'].apply(lambda x: char_rating(x))

        # GROUPING THE TWEETS BEFORE AND THE TWEETS AFTER THE GAME
        tweets_before = tweets[tweets['when'] == 'before'].groupby(['player_name', 'game_date'])[
            ['vader_polarity', 'tb_polarity']].mean()
        tweets_after = tweets[tweets['when'] == 'after'].groupby(['player_name', 'game_date'])[
            ['vader_polarity', 'tb_polarity']].mean()

        # MERGE WITH PREMIER CSV
        sent_before = pd.merge(premier, tweets_before, how='outer', on=('player_name', 'game_date'))
        sent_before['vader_emotion'] = sent_before['vader_polarity'].apply(lambda x: emotion(x))
        sent_before['tb_emotion'] = sent_before['tb_polarity'].apply(lambda x: emotion(x))

        sent_after = pd.merge(premier, tweets_after, how='outer', on=('player_name', 'game_date'))
        sent_after['vader_emotion'] = sent_after['vader_polarity'].apply(lambda x: emotion(x))
        sent_after['tb_emotion'] = sent_after['tb_polarity'].apply(lambda x: emotion(x))

        # FINAL MERGE
        sentiment_full = pd.merge(sent_before, sent_after, how='outer',
                                  on=('team_id', 'team_name', 'player_id', 'player_name', 'nationality',
                                      'age', 'primary_pos', 'game_date', 'rating', 'minutes', 'goal', 'assist',
                                      'char_rating'), suffixes=('_before', '_after'))

        # RESULTS
        print(tabulate(sentiment_full, headers='keys'))
        print("Dataframe keys are:", sentiment_full.keys())

        # WRITING THE CSV
        sentiment_full.to_csv('premier_with_sentiment.csv', index=False)
        print(f"\n\033[32mCOMPILED CSV FILE AVAILABLE: premier_with_sentiment.csv\033[0m")

if __name__ == "__main__":
    merge_premier_tweets()