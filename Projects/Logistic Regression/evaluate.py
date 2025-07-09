import numpy as np

def recall(y_hat, y):
    # Computes the number of songs in the playlist
    # that were correctly predicted to belong in the playlist.
    # Eg. Out of all the songs in the rap playlist, how many did get right.
    # 'y': array of labels
    # 'y_hat': array of predictions
    
    in_playlist_idx = np.where(y==1)[0]
    labels = y[in_playlist_idx]
    predictions = y_hat[in_playlist_idx]
    num_correct = sum(labels == predictions)
    total = len(labels)
    return num_correct, total

def precision(y_hat, y):
    # Computes the number songs predicted to be in the playlist
    # that were actually in the playlist.
    # Eg. Out of all the songs we predicted to be rap, how many were actually rap?
    # 'y': array of labels
    # 'y_hat': array of predictions

    predicted_in_playlist_idx = np.where(y_hat==1)[0]
    labels = y[predicted_in_playlist_idx]
    predictions = y_hat[predicted_in_playlist_idx]
    num_correct = sum(labels == predictions)
    total = len(labels)
    return num_correct, total
    
def evaluate(y_hat, y, quiet=False, return_all=False):
    # Prints summary of performance
    num_correct = sum(y_hat == y)
    total = len(y)
    acc = num_correct/total
    
    rec, rec_total = recall(y_hat, y)
    rec_pct = rec/rec_total
        

    prec, prec_total = precision(y_hat, y)
    if prec_total != 0:
        prec_pct = prec/prec_total
    else:
        prec_pct = 1

    f1 = (2 * prec_pct * rec_pct)/(prec_pct + rec_pct)
    if not quiet:
        print(f"Accuracy: {round(100*acc, 2)}% ({num_correct}/{total})")
        print(f"Num Predicted in Playlist/Num In playlist: {round(100*rec_pct, 2)}% ({rec}/{rec_total})")
        if prec_total != 0:
            print(f"Num in Playlist/Num Predicted In playlist: {round(100*prec_pct, 2)}% ({prec}/{prec_total})")
        else:
            print(f"Num in Playlist/Num Predicted In playlist: 100% (0/0)")
        print(f"F1 Score: {round(100*f1, 2)}%")
    if not return_all:
        return f1
    else:
        return num_correct, total, rec, rec_total, prec, prec_total, f1