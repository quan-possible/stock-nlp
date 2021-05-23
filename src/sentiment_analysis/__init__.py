import random
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import transformers

def get_pred(classifier, text, length, batch_size=100,
             sent_dict={"LABEL_0":-1, "LABEL_1":0, "LABEL_2":1}):
    
    res = np.zeros((length,2))
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

    a = max(0,length-1)
    assert round(classifier(text[a])[0]['score'],2) \
        == round(pred[a,1],2), f"Failed on index {a}"

    a = max(0, length-2)
    assert round(classifier(text[a])[0]['score'],2) \
        == round(pred[a,1],2), f"Failed on index {a}"

    print("Test succeeded!")

if __name__ == "__main__":

    parser = ArgumentParser(description="Sentiment analysis given tweets.")
    parser.add_argument("--pretrained_model", type=str, 
        default="cardiffnlp/twitter-roberta-base-sentiment",
        help="Pretrained model for sentiment analysis.")
    parser.add_argument("--test_size", type=int, 
        default=10000, help="Choose size of data for unit test.")
    parser.add_argument("--batch_size", type=int, 
        default=100, help="Size of batches given to classifier.")
    parser.add_argument("--data_path", type=str, 
        default="project/data/processed.csv", help="Path to target data.")
    parser.add_argument("--save_path", type=str, 
        default="project/data/tagged.csv", help="Path for data saving.")
    parser.add_argument("--device", type=int, 
        default=0, help="Choose to use a cpu or gpu.")

    args = parser.parse_args()

    batch_size = args.batch_size
    # ------------
    # data
    # ------------
    df = pd.read_csv(args.data_path, parse_dates=[3])

    # ------------
    # model
    # ------------
    classifier = transformers.pipeline("sentiment-analysis",
        device=args.device, model=args.model, binary_output=True
    )

    # Unit test
    test_pred(df.Tweet, args.test_size, args.batch_size)

    # ------------
    # predict
    # ------------
    res = get_pred(df.Tweet, len(df.Tweet), args.batch_size)
    df_res = pd.DataFrame(res, columns=["Sentiment","Score"])

    # Save result
    df2 = pd.concat([df,df_res], axis = 1)
    df2.to_csv(args.save_path, index=False)