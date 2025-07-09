########################
### Import Data
########################

Spotify <- read.csv("Spotify.csv")

########################
### Import Libraries
########################

library(dplyr)
library(stringi)
library(lubridate)
library(music)

###################
### Featurize Data
###################

# Add indicator for each playlist then group by song
# and for each indicator take the max of the indicators
# in order to account for playlists that are in multiple
# playlists. Then keep only one entry per song
Spotify <- Spotify %>%
  mutate(in_Blend = as.integer(Playlist == "Blend"),
         in_Bliss = as.integer(Playlist == "Bliss"),
         in_Rock = as.integer(Playlist == "Rock"),
         in_Rap = as.integer(Playlist == "Rap"),
         in_Oldies = as.integer(Playlist == "Oldies"),
         in_Orchestral = as.integer(Playlist == "Orchestral"),
         in_Cinema = as.integer(Playlist == "Cinema")) %>%
  group_by(Song) %>%
  mutate(in_Blend = max(in_Blend),
         in_Bliss = max(in_Bliss),
         in_Rock = max(in_Rock),
         in_Rap = max(in_Rap),
         in_Oldies = max(in_Oldies),
         in_Orchestral = max(in_Orchestral),
         in_Cinema = max(in_Cinema)
  ) %>%
  ungroup() %>%
  distinct(Song, .keep_all = TRUE) %>%
  subset(select = -c(Playlist, Playlist.ID))

# Separate Key Signature into Key and Modality
Spotify <- Spotify %>%
  mutate(Mode = stri_reverse(substring(stri_reverse(Key.Signature.Formatted), 1, 3)),
         Key = gsub(" ", "", substring(Key.Signature.Formatted, 1, 2))
         ) %>%
  subset(select = -Key.Signature.Formatted)

# One hot encoding for Time Signature (if treating as categorical)
# NOTE: we need to disregard one value so as not to have perfect multicollinearity
# We are still keeping the original Time Signature column in case
# we want to treat it as a numerical column
time_sigs <- unique(Spotify$Time.Signature)[-1]

for (time_sig in time_sigs){
  Spotify <- Spotify %>%
    mutate(!!paste0("TimeSignature_", time_sig) := as.integer(Time.Signature == time_sig))
}

###################
### Fix Data Types
###################

# Change True/False to indicators
Spotify <- Spotify %>%
  mutate(Explicit = as.integer(Explicit == "True"))

# Change release date to Date object then get age
# Note if only year is provided we shall assume it was released Jan. 1
#current_date <- Sys.Date()
current_date <- as.Date("2025-05-11")

Spotify <- Spotify %>%
  mutate(Release.Date = substring(ifelse(is.na(as.Date(Release.Date)),
                                         paste0(Release.Date, "-01-01"),
                                         Release.Date
                                         ),
                                  1, 10
                                  ),
         Age = interval(Release.Date, current_date) %/% months(1)
         ) %>%
  subset(select = -Release.Date)

# Change Modality to indicator of Major
Spotify <- Spotify %>%
  mutate(Mode = as.integer(Mode == "Maj"))

# Change Key to frequency of note in Octave 4 (Hz)
Spotify <- Spotify %>%
  mutate(Key = note2freq(paste0(Key, "4")))


###################
### Write to new csv
###################

write.csv(Spotify, "Spotify_clean.csv", row.names = FALSE)