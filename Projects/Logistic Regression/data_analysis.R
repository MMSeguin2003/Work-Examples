###########################
### Import Cleaned Data ###
###########################

Spotify <- read.csv("Spotify_clean.csv")

########################
### Import Libraries ###
########################

library(dplyr)
library(ggplot2)



#############################################
#### Graph Function for General Playlist ####
#############################################

categorical_columns <- c("Explicit", "Mode", "Time.Signature", "Key")

plot_playlist <- function(playlist, covariate, type = NA){
  # Use bar chart for categorical variables
  if (covariate %in% categorical_columns){
    plot <- Spotify %>%
      mutate(!!covariate := as.factor(!!sym(covariate)),
             !!paste0("in_", playlist) := as.factor(!!sym(paste0("in_", playlist)))
             ) %>%
      ggplot(aes(x = !!sym(covariate),
                 fill = !!sym(paste0("in_", playlist))
                 )
             ) +
      geom_bar(position = "fill") +
      scale_x_discrete(labels = function(x) format(round(as.numeric(x), 1), nsmall = 1)) +
      labs(x = covariate,
           y = "Count",
           fill = paste0("In ", playlist, "\nPlaylist"),
           title = paste(covariate, "effect on presence in", playlist)
           ) +
      theme_bw()
  # Use boxplot, histogram, or density for numerical variables
  } else {
    if (type == "boxplot"){
      plot <- Spotify %>%
        mutate(!!paste0("in_", playlist) := as.factor(!!sym(paste0("in_", playlist)))
               ) %>%
        ggplot(aes(x = !!sym(paste0("in_", playlist)),
                   y = !!sym(covariate)
                   )
               ) +
        geom_boxplot() +
        labs(x = paste0("In ", playlist, " Playlist"),
             y = covariate,
             title = paste(covariate, "effect on presence in", playlist)
             ) +
        theme_bw()
    } else if (type == "histogram"){
      plot <- Spotify %>%
        mutate(!!paste0("in_", playlist) := as.factor(!!sym(paste0("in_", playlist)))
               ) %>%
        ggplot(aes(x = !!sym(covariate),
                   y = after_stat(density),
                   fill = !!sym(paste0("in_", playlist))
                   )
               ) +
        geom_histogram(position = "identity",
                       bins = 50,
                       alpha = 0.75) +
        labs(x = covariate,
             y = "Density",
             fill = paste0("In ", playlist, "\nPlaylist"),
             title = paste(covariate, "effect on presence in", playlist)
             ) +
        theme_bw()
    } else if (type == "density"){
      plot <- Spotify %>%
        mutate(!!paste0("in_", playlist) := as.factor(!!sym(paste0("in_", playlist)))
               ) %>%
        ggplot(aes(x = !!sym(covariate),
                   y = after_stat(density),
                   fill = !!sym(paste0("in_", playlist))
                   )
               ) +
        geom_density(position = "identity",
                     alpha = 0.75) +
        labs(x = covariate,
             y = "Density",
             fill = paste0("In ", playlist, "\nPlaylist"),
             title = paste(covariate, "effect on presence in", playlist)
             ) +
        theme_bw()
    } else {
      return(NA)
    }
  }
  return(plot)
}



##################################################################
#######                 Graphical Analysis                 #######
##################################################################
#######                        Notes                       #######
#
# When looking for good predictor variables we want the distributions
# to be different for the different values of the indicator. Differing
# distributions of the predictor for each value of the indicator shows
# that certain values of that predictor correspond more to one value
# of the indicator than the other. For categorical predictors we will
# use bar charts to compare the distributions and for numerical
# predictors we will use histograms to compare the distributions.
#
# To denote possible predictor variables we will use # to indicate
# it is potentially a good predictor, ## to indicate it is potentially
# a very good predictor, and ## to indicate it is an excellent predictor
#
# We will now continue to look at each playlist of interest

###########################
####       Blend       ####
###########################

###########################
### Categorical Columns ###
###########################

plot_playlist("Blend", "Explicit") #
plot_playlist("Blend", "Mode")
plot_playlist("Blend", "Time.Signature")
plot_playlist("Blend", "Key")

###########################
###  Numerical Columns  ###
###########################

plot_playlist("Blend", "Playtime", "histogram")
plot_playlist("Blend", "Loudness", "histogram") #
plot_playlist("Blend", "Popularity", "histogram")
plot_playlist("Blend", "Energy", "histogram") ##
plot_playlist("Blend", "Positiveness", "histogram") ##
plot_playlist("Blend", "Speechiness", "histogram")
plot_playlist("Blend", "Liveliness", "histogram")
plot_playlist("Blend", "Acousticness", "histogram") #
plot_playlist("Blend", "Instrumentalness", "histogram") ##
plot_playlist("Blend", "Danceability", "histogram") #
plot_playlist("Blend", "Tempo", "histogram")
plot_playlist("Blend", "Age", "histogram")

###########################
####       Bliss       ####
###########################

###########################
### Categorical Columns ###
###########################

plot_playlist("Bliss", "Explicit") #
plot_playlist("Bliss", "Mode")
plot_playlist("Bliss", "Time.Signature")
plot_playlist("Bliss", "Key")

###########################
###  Numerical Columns  ###
###########################

plot_playlist("Bliss", "Playtime", "histogram")
plot_playlist("Bliss", "Loudness", "histogram")
plot_playlist("Bliss", "Popularity", "histogram")
plot_playlist("Bliss", "Energy", "histogram") #
plot_playlist("Bliss", "Positiveness", "histogram") #
plot_playlist("Bliss", "Speechiness", "histogram") #
plot_playlist("Bliss", "Liveliness", "histogram")
plot_playlist("Bliss", "Acousticness", "histogram") #
plot_playlist("Bliss", "Instrumentalness", "histogram") ##
plot_playlist("Bliss", "Danceability", "histogram") #
plot_playlist("Bliss", "Tempo", "histogram")
plot_playlist("Bliss", "Age", "histogram")

###########################
####       Rock        ####
###########################

###########################
### Categorical Columns ###
###########################

plot_playlist("Rock", "Explicit") ##
plot_playlist("Rock", "Mode")
plot_playlist("Rock", "Time.Signature")
plot_playlist("Rock", "Key")

###########################
###  Numerical Columns  ###
###########################

plot_playlist("Rock", "Playtime", "histogram")
plot_playlist("Rock", "Loudness", "histogram") ##
plot_playlist("Rock", "Popularity", "histogram") #
plot_playlist("Rock", "Energy", "histogram") ##
plot_playlist("Rock", "Positiveness", "histogram") ##
plot_playlist("Rock", "Speechiness", "histogram") #
plot_playlist("Rock", "Liveliness", "histogram")
plot_playlist("Rock", "Acousticness", "histogram") ##
plot_playlist("Rock", "Instrumentalness", "histogram") ##
plot_playlist("Rock", "Danceability", "histogram") #
plot_playlist("Rock", "Tempo", "histogram") #
plot_playlist("Rock", "Age", "histogram") #

###########################
####       Rap         ####
###########################

###########################
### Categorical Columns ###
###########################

plot_playlist("Rap", "Explicit") ###
plot_playlist("Rap", "Mode")
plot_playlist("Rap", "Time.Signature") #
plot_playlist("Rap", "Key")

###########################
###  Numerical Columns  ###
###########################

plot_playlist("Rap", "Playtime", "histogram")
plot_playlist("Rap", "Loudness", "histogram") ##
plot_playlist("Rap", "Popularity", "histogram") #
plot_playlist("Rap", "Energy", "histogram") #
plot_playlist("Rap", "Positiveness", "histogram") #
plot_playlist("Rap", "Speechiness", "histogram") ###
plot_playlist("Rap", "Liveliness", "histogram")
plot_playlist("Rap", "Acousticness", "histogram") #
plot_playlist("Rap", "Instrumentalness", "histogram") ###
plot_playlist("Rap", "Danceability", "histogram") ##
plot_playlist("Rap", "Tempo", "histogram")
plot_playlist("Rap", "Age", "histogram") #

###########################
####     Oldies        ####
###########################

###########################
### Categorical Columns ###
###########################

plot_playlist("Oldies", "Explicit") ###
plot_playlist("Oldies", "Mode")
plot_playlist("Oldies", "Time.Signature")
plot_playlist("Oldies", "Key")

###########################
###  Numerical Columns  ###
###########################

plot_playlist("Oldies", "Playtime", "histogram") ##
plot_playlist("Oldies", "Loudness", "histogram")
plot_playlist("Oldies", "Popularity", "histogram")
plot_playlist("Oldies", "Energy", "histogram")
plot_playlist("Oldies", "Positiveness", "histogram")
plot_playlist("Oldies", "Speechiness", "histogram") #
plot_playlist("Oldies", "Liveliness", "histogram")
plot_playlist("Oldies", "Acousticness", "histogram") ##
plot_playlist("Oldies", "Instrumentalness", "histogram") ##
plot_playlist("Oldies", "Danceability", "histogram")
plot_playlist("Oldies", "Tempo", "histogram")
plot_playlist("Oldies", "Age", "histogram")

###########################
###    Orchestral      ####
###########################

###########################
### Categorical Columns ###
###########################

plot_playlist("Orchestral", "Explicit") ###
plot_playlist("Orchestral", "Mode")
plot_playlist("Orchestral", "Time.Signature")
plot_playlist("Orchestral", "Key")

###########################
###  Numerical Columns  ###
###########################

plot_playlist("Orchestral", "Playtime", "histogram")
plot_playlist("Orchestral", "Loudness", "histogram") ##
plot_playlist("Orchestral", "Popularity", "histogram")
plot_playlist("Orchestral", "Energy", "histogram") ##
plot_playlist("Orchestral", "Positiveness", "histogram") ##
plot_playlist("Orchestral", "Speechiness", "histogram") #
plot_playlist("Orchestral", "Liveliness", "histogram")
plot_playlist("Orchestral", "Acousticness", "histogram") ###
plot_playlist("Orchestral", "Instrumentalness", "histogram") ###
plot_playlist("Orchestral", "Danceability", "histogram") #
plot_playlist("Orchestral", "Tempo", "histogram") #
plot_playlist("Orchestral", "Age", "histogram")

###########################
###       Cinema       ####
###########################

###########################
### Categorical Columns ###
###########################

plot_playlist("Cinema", "Explicit") ###
plot_playlist("Cinema", "Mode")
plot_playlist("Cinema", "Time.Signature")
plot_playlist("Cinema", "Key")

###########################
###  Numerical Columns  ###
###########################

plot_playlist("Cinema", "Playtime", "histogram")
plot_playlist("Cinema", "Loudness", "histogram") #
plot_playlist("Cinema", "Popularity", "histogram")
plot_playlist("Cinema", "Energy", "histogram") #
plot_playlist("Cinema", "Positiveness", "histogram") ##
plot_playlist("Cinema", "Speechiness", "histogram") #
plot_playlist("Cinema", "Liveliness", "histogram")
plot_playlist("Cinema", "Acousticness", "histogram") ##
plot_playlist("Cinema", "Instrumentalness", "histogram") ##
plot_playlist("Cinema", "Danceability", "histogram") ##
plot_playlist("Cinema", "Tempo", "histogram")
plot_playlist("Cinema", "Age", "histogram")