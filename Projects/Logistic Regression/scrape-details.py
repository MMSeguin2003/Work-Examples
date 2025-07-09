import re
import os
import ast
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from seleniumbase import Driver
from bs4 import BeautifulSoup as bs

load_dotenv();
AGENT = os.getenv("AGENT")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
USER = os.getenv("USER")
PASS = os.getenv("PASS")
BASE_URL = os.getenv("BASE_URL")
PROXY = f"{USER}:{PASS}@{HOST}:{PORT}"

with open("songs.txt", "r") as f:
    songs = ast.literal_eval(f.read())
n = len(songs["Song"])

def get_search_url(song):
    # Get song name and fitler out special characters
    song_name = song[0]
    song_name = re.sub("-|\\(|\\)|'|\\?|\\.|,|\\\\", "", song_name)
    song_name = re.sub("\\s+", "-", song_name).lower()
    # Get song ID
    song_id = song[1]
    # Create query string a nd url
    query = "/track/" + song_name + "/" + song_id
    url = BASE_URL + query
    return(url)

driver = Driver(uc = True,
                    headless = False,
                    no_sandbox = True,
                    disable_gpu = True,
                    ad_block = True,
                    agent = AGENT,
                    proxy = PROXY)

def scrape_song_details(song):
    details = {}

    # Get url and page html content
    url = get_search_url(song)
    driver.uc_open_with_reconnect(url, reconnect_time = 6)
    driver.uc_gui_click_captcha()
    soup = driver.get_page_source()
    soup = bs(soup, "html.parser")

    # Get elements containing song details
    facts_containers = soup.find_all("div", class_ = "song-fact-container-stat")
    stat_containers = soup.find_all("div", class_ = "song-bar-statistic-number")

    # Get details and add to dictionary
    details["Tempo"] = facts_containers[0].contents[0].strip()
    details["Tempo"] = int(re.sub("[^0-9]", "", details["Tempo"]))
    details["Key Signature Formatted"] = facts_containers[1].contents[0].strip()
    details["Loudness"] = facts_containers[2].contents[0].strip()
    details["Loudness"] = float(re.sub("b|d", "", details["Loudness"]))
    details["Time Signature"] = facts_containers[3].contents[0].strip()
    details["Energy"] = stat_containers[1].contents[0].strip()
    details["Energy"] = int(re.sub("%", "", details["Energy"]))
    details["Danceability"] = stat_containers[2].contents[0].strip()
    details["Danceability"] = int(re.sub("%", "", details["Danceability"]))
    details["Positiveness"] = stat_containers[3].contents[0].strip()
    details["Positiveness"] = int(re.sub("%", "", details["Positiveness"]))
    details["Speechiness"] = stat_containers[4].contents[0].strip()
    details["Speechiness"] = int(re.sub("%", "", details["Speechiness"]))
    details["Liveliness"] = stat_containers[5].contents[0].strip()
    details["Liveliness"] = int(re.sub("%", "", details["Liveliness"]))
    details["Acousticness"] = stat_containers[6].contents[0].strip()
    details["Acousticness"] = int(re.sub("%", "", details["Acousticness"]))
    details["Instrumentalness"] = stat_containers[7].contents[0].strip()
    details["Instrumentalness"] = int(re.sub("%", "", details["Instrumentalness"]))
    return(details)

for j in range(n):
    # Check if already have data
    if songs["Instrumentalness"][j] == "":
        s = [songs["Song"][j], songs["Song ID"][j]]
        # Attempt webscrape
        try:
            details = scrape_song_details(s)
        # If something happens wait 60 seconds
        except:
            time.sleep(60)
            # Try again
            try:
                details = scrape_song_details(s)
            # If failed again make results NA
            except:
                details = {}
        for key in details:
            if key in songs:
                songs[key][j] = details[key]
        # Add nonconstant delay between requests
        time.sleep(np.random.uniform(0.05, 0.75, 1)[0])

songs[["Key", "Mode"]] = songs["Key Signature Formatted"].str.split(" ", expand = True)

songs = pd.DataFrame(songs)

songs.to_csv("Spotify.csv", index = False)