import nltk
import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize
from urllib.request import urlopen
import re
from tqdm import tqdm
import csv
from datetime import datetime, timedelta
from convertions import convert_date, months
from csv_management import write_occurrences_csv

def get_teams(url):
    """
        Collects the url (id/team_name) for every team in the league
    """

    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    teams_soup = re.findall('/teams/' + "(.*?)" + '",', str(soup))

    teams = []
    for team in tuple(teams_soup):
        info = str(team).replace("overview/", "")
        if info not in teams:
            teams.append(info)

    return teams

def get_players(team):
    """
        Collects the url (id/name) for every player in the team
    """

    team_id = str(team).split("/")[0]
    team_name = str(team).split("/")[1]
    url = "https://www.fotmob.com/teams/"+ team_id + "/squad/" + team_name
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    athletes = re.findall('"athlete":' + "(.*?)" + '}}],"location":{"@type":"Place"', str(soup))
    players = re.findall('"https://www.fotmob.com/players/' + "(.*?)" + '","nationality":', str(athletes))
    players = players[0:int((len(players)/2))]

    return players

def get_informations(info, output, team_name, team_id, lost_players, missing_players):
    """
        Collects the informations for the given player and returns a tuple
    """

    try:
        url = "https://www.fotmob.com/players/" + str(info)

        html = requests.get(url).content
        soup = BeautifulSoup(html, features="html.parser")

        age_nat = get_player_age_nat(soup)

        primary_pos = get_player_position(soup)

        part = soup.find_all("tr", {
            "class": "css-ec7htm-PlayerCareerMatchStatsBodyCSS-playerCareerMatchStatsHeaderCommon e1b3vhfl9"})
        if len(part) == 0:
            part = soup.find_all("tr", {"class": "css-sb1lc8-PlayerCareerMatchStatsBodyCSS-playerCareerMatchStatsHeaderCommon e1b3vhfl9"})
            if len(part) == 0:
                missing_players.append(info)

        name = get_player_name(soup)
        id = get_player_id(info)

        for block in str(part).split('PlayerCareerMatchStatsBodyCSS'):
            date = re.findall('<td><span>' + "(.*?)" + '</span></td', str(block))
            grade = re.findall('"><span>' + "(.*?)" + '</span></div>', str(block))
            other = re.findall('e1b3vhfl1">' + "(.*?)" + '</span></td><td><span', str(block))

            if not grade:
                rating = "NA"
            if grade:
                rating = grade[0]

            if date:
                output.append((team_id, team_name, id, name, age_nat[1], age_nat[0], primary_pos, convert_date(date[0], months), rating, other[0], other[1], other[2]))
                # team id, team name, player id, player name, nationality, age, primary position, date, rating, minutes, goal, assist
    except:
        lost_players.append(info)

def get_player_name(soup):
    """
        Collects the name and the lastname of the player
    """

    part = soup.find_all("h1", {
        "class": "css-1x7my78-PlayerName e1258a1r2"})
    name = re.findall('-PlayerName e1258a1r2">' + "(.*?)" + '</h1>', str(part))

    return str(name[0]).upper()

def get_player_id(url):
    """
        Collects the id of the player
    """
    return str(url).split("/")[0]

def get_player_age_nat(soup):
    try:
        part_basics = soup.find_all("b", {"class": "css-u48vbb-StatValue e12dujii2"})
        basics = re.findall('e12dujii2">' + "(.*?)" + '</b>', str(part_basics))

        if len(basics) == 6:
            age = basics[2]
            nationality = basics[3]
        elif len(basics) == 5:
            age = basics[1]
            nationality = basics[2]
        elif len(basics) == 4:
            age = basics[1]
            nationality = basics[2]
        elif len(basics) == 3:
            age = basics[0]
            nationality = basics[1]
        elif len(basics) == 2:
            age = basics[0]
            nationality = basics[1]
        elif len(basics) < 2:
            age = 'NA'
            nationality = 'NA'
    except:
        age = 'NA'
        nationality = 'NA'

    return (age, nationality)

def get_player_position(soup):
    """
        Collects the position of the player (english acronym)
    """
    primary_pos = 'NA'
    try:
        part = soup.find_all("span", {"class": "css-reooou-PositionsText e1vv4hxj0"})
        pos = re.findall('e1vv4hxj0">' + "(.*?)" + '</span>', str(part))
        if len(pos[0].split()) > 1:
            primary_pos = pos[0].split()[0][0] + pos[0].split()[1][0]
        elif len(pos[0].split()) == 1:
            if len(pos[0]) > 2:
                primary_pos = pos[0][0]
            else:
                primary_pos = pos[0]
        else:
            primary_pos = 'NA'
    except:
        primary_pos = 'NA'

    return primary_pos

def get_team_name(id_name):
    """
        Collects the team name
    """
    return str(id_name).split('/')[1].upper()

def get_team_id(id_name):
    """
        Collects the team id
    """
    return str(id_name).split('/')[0].upper()

def collec_players_informations(url):
    """
        Collects the informations for every player in the league
    """

    occurrences = []
    missing_players = []
    lost_players = []

    league_name = url.split("/")[-1]
    print(f"\nLEAGUE NAME: {league_name}")

    teams = get_teams(url)
    print(f"\nCOLLECTED TEAMS: {teams}")

    team_n = len(teams)
    current = 1

    for team in teams:
        informations = get_players(team)
        team_name = get_team_name(team)
        team_id = get_team_id(team)

        print(f"\n\033[36mGETTING INFORMATIONS ABOUT {team_name} PLAYERS ({current}/{team_n})\033[0m")

        for player in tqdm(informations):
            get_informations(player, occurrences, team_name, team_id, lost_players, missing_players)

        current += 1

    print(f"\n\033[31mPROCESS FINISHED WITH {len(lost_players)} ERRORS\nLOST PLAYERS ARE: {lost_players} \033[0m")
    print(f"\n\033[33mINFORMATIONS NOT AVAILABLE FOR {len(missing_players)} PLAYERS\nMISSING PLAYERS ARE: {missing_players} \033[0m")
    print(f"\n\033[32mOCCURRENCES NUMBER: {len(occurrences)}\033[0m")

    return (league_name, occurrences)

if __name__ == "__main__":
    url = "https://www.fotmob.com/leagues/47/overview/premier-league"
    to_write = collec_players_informations(url)
    write_occurrences_csv(to_write[0], to_write[1])