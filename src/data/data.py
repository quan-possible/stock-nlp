import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    DATA_PATH = 'project/data/training_set_tweets.txt'
    SAVE_PATH = 'project/data/processed.csv'

    df = pd.read_csv(
        DATA_PATH, delimiter = "\t",
        names=["UserID", "TweetID", "Tweet", "CreatedAt"],
        error_bad_lines=False, warn_bad_lines=False
    )

    # coerce the IDs columns to int64
    df["UserID"] = pd.to_numeric(df["UserID"], errors="coerce")
    df["TweetID"] = pd.to_numeric(df["TweetID"], errors="coerce")
    df[["UserID","TweetID"]] = df[["UserID","TweetID"]].astype('int64', errors='ignore')

    # coerce the CreatedAt column into rounded datetime
    df["CreatedAt"] = pd.to_datetime(df["CreatedAt"],dayfirst=True, errors='coerce')
    df.CreatedAt = df.CreatedAt.dt.round('H')

    df.to_csv(SAVE_PATH, index = False)