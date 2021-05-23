---

<div align="center">    
 
# Intraday Market Movement Prediction Using Sentiment Analysis and Historical Data
 <img src="https://github.com/quan-possible/stock-nlp/blob/master/documents/stonks.jpg" alt="drawing" width="400"/>
 
 
 
 Read our literature review: https://github.com/quan-possible/stock-nlp/blob/master/documents/literature_review.pdf
 
 Read our paper: https://github.com/quan-possible/stock-nlp/blob/master/documents/paper.pdf
 
## Description   
In this project, we created a model that predict the intraday movement of the S&P 500 Index. It uses both the historical index value and public sentiments derived from Tweets. The result is a satisfactory accuracy level.

## Datasets

- Data used for the International Workshop on Semantic Evaluation (SemEval): SemEval is an periodical series of NLP workshops with a mission to advance the current state of the art in semantic analysis and to help create high-quality annotated datasets for tasks in natural language semantics. We use the data from SemEval-2017 in particular to train the sentiment analysis model.

- Cheng-Caverlee-Lee dataset: A collection of over 9 million public tweets geo-located in the United States. It is used to generate public sentiments which is then used for market movement prediction.

## Methods

- Sentiment Analysis: Word Count (baseline), Decision tree (baseline) and RoBERTa.

- Market movement prediction: SVM (baseline) and GRUs.





