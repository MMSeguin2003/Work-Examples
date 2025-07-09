import os
import re
import time
import spotipy
from dotenv import load_dotenv
from unidecode import unidecode
from spotipy.oauth2 import SpotifyOAuth

# Get authorization details and access API
load_dotenv();
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

SCOPE = "user-library-read playlist-read-private"

sp = spotipy.Spotify(auth_manager = SpotifyOAuth(client_id = CLIENT_ID,
                                                 client_secret = CLIENT_SECRET,
                                                 redirect_uri = REDIRECT_URI,
                                                 scope = SCOPE)
                    )

# Create function to get all playlists
def get_playlists():
    playlists = {}
    results = sp.current_user_playlists()
    
    while results:
        for item in results["items"]:
            playlists[item["name"]] = item["id"]
        if results["next"]:
            results = sp.next(results)
        else:
            break
    
    return(playlists)

# Get all playlists and intialize dictionary of songs
playlists = get_playlists()
songs = {
    "Song": [],
    "Song ID": [],
    "Artist": [],
    "Release Date": [],
    "Album": [],
    "Explicit": [],
    "Playlist": [],
    "Playlist ID": [],
    "Playtime": [],
    "BPM": [],
    "Key Signature": [],
    "Loudness": [],
    "Time Signature": [],
    "Popularity": [],
    "Energy": [],
    "Danceability": [],
    "Positiveness": [],
    "Speechiness": [],
    "Liveliness": [],
    "Acousticness": [],
    "Instrumentalness": []
    }


# Create function to get all songs
def get_songs(playlist):
    songs = []
    song_ids = []
    results = sp.playlist_tracks(playlist[1])

    while results:
        for item in results["items"]:
            track = item["track"]
            songs.append(track["name"])
            song_ids.append(track["id"])
        if results["next"]:
            results = sp.next(results)
        else:
            break

    return(songs, song_ids)

# Get all songs
for playlist in playlists.items():
    s, s_ids = get_songs(playlist)
    songs["Song"].extend(s)
    songs["Song ID"].extend(s_ids)
    songs["Playlist"].extend([playlist[0] for _ in range(len(s))])
    songs["Playlist ID"].extend([playlist[1] for _ in range(len(s))])

def get_song_details(song):
    details = {}

    # Get details and add to dictionary
    track = sp.track(song[1])
    details["Playtime"] = track["duration_ms"]
    details["Popularity"] = track["popularity"]
    details["Explicit"] = track["explicit"]
    details["Artist"] = track["artists"][0]["name"]
    details["Album"] = track["album"]["name"]
    details["Release Date"] = track["album"]["release_date"]
    return(details)

for j in range(len(songs["Song"])):
    s = [songs["Song"][j], songs["Song ID"][j]]
    details = get_song_details(s)
    for key in details:
        if key in songs:
            songs[key].append(details[key])
    time.sleep(0.5)

# Filter out non-standard characters
for key in songs:
    for j in range(len(songs[key])):
        if isinstance(songs[key][j], str):
            songs[key][j] = unidecode(songs[key][j])
            songs[key][j] = songs[key][j].replace("\u0159", "r")

n = len(songs["Song"])
for key in songs:
    m = len(songs[key])
    if m < n:
        songs[key].extend(["" for _ in range(n - m)])

with open("songs.txt", "w") as f:
    f.write(str(songs))