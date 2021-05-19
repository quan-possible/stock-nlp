import random
import time
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import transformers

from sklearn.metrics import accuracy_score,confusion_matrix

def get_pred(classifier, text, length, batch_size=100,
             sent_dict={"LABEL_0":-1, "LABEL_1":0, "LABEL_2":1}):
    
    res = np.zeros((length,2))
    res[:] = np.nan
    cur_row = 0
    loading = 0

    t_start = time.process_time_ns()
    print(f"Row: {cur_row}. Progress: {loading}/100")
    for row in range(0,length,batch_size):
        output = classifier(list(text[row:min(length,row+batch_size)]))
        for i,elem in enumerate(output):
            senti = sent_dict[elem["label"]]
            score = elem["score"]

            cur_row = row + i
            res[cur_row] = [senti, score]

        if round(cur_row/length*100,0) > loading:

            loading = round(cur_row/length*100,0)
            t_stop = time.process_time_ns()
            elapsed_time = t_stop - t_start
            time_left = ((100-loading) * elapsed_time)*1e-9
            mins = round(time_left/60, 0)
            secs = round(time_left%60, 0)

            
            print(f"Row: {cur_row}. Progress: {loading}/100. \
                Time left: {mins} mins {secs} secs")

            t_start = time.process_time_ns()

    print("Processing complete!")
    return res


def test_pred(classifier, text, length, batch_size):

    pred = get_pred(text, length, batch_size)

    a = random.randint(0,length)
    assert round(classifier(text[a])[0]['score'],2) \
        == round(pred[a,1],2), f"Failed on index {a}"

    a = random.randint(0,length)
    assert round(classifier(text[a])[0]['score'],2) \
        == round(pred[a,1],2), f"Failed on index {a}"
    
    a = 0
    assert round(classifier(text[a])[0]['score'],2) \
        == round(pred[a,1],2), f"Failed on index {a}"

    a = length-1
    assert round(classifier(text[a])[0]['score'],2) \
        == round(pred[a,1],2), f"Failed on index {a}"

    a = max(0, length-2)
    assert round(classifier(text[a])[0]['score'],2) \
        == round(pred[a,1],2), f"Failed on index {a}"

    print("Test succeeded!")

if __name__ == "__main__":

    TEST_SIZE = 10000
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    DATA_PATH = "project/data/processed.csv"
    SAVE_PATH = "project/data/tagged.csv"
    DEVICE = 0

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    os.chdir("..")
    batch_size = args.batch_size
    
    df = pd.read_csv(DATA_PATH, parse_dates=[3])

    # Get classifier
    classifier = transformers.pipeline("sentiment-analysis",
        device=DEVICE, model=MODEL, binary_output=True
    )

    # Unit test
    test_pred(df.Tweet, TEST_SIZE, batch_size)

    # Get prediction
    res = get_pred(df.Tweet, len(df.Tweet), batch_size)
    df_res = pd.DataFrame(res, columns=["Sentiment","Score"])

    # Save result
    df2 = pd.concat([df,df_res], axis = 1)
    df2.to_csv(SAVE_PATH, index=False)

