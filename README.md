# Multi-Turn Response Selection

This repository contains code for the paper "[Memory-Based Matching Models for Multi-turn Response Selection in Retrieval-Based Chatbots](http://tcci.ccf.org.cn/conference/2018/papers/EV43.pdf)".

## About this code

This is the code by ECNU submitted to nlpcc2018 Task5 sub1. The MBMN(MVM)+SMN+NLP model achieves 62.61% Precision score on the test set and ranks 1st among all the participants. 

## Installation

```
# download the repo
git clone https://github.com/gongwu/nlpcc2018-Task5-MultiTurnResponseSelection.git
# download the dataset
# run the model
python main/run_dialogue_SCNRMA.py
```

## Results on Dev

|                |       Model        | Precision (%) |
| :------------: | :----------------: | :-----------: |
|                |    NLP features    |     39.67     |
|                |   SMN [ACL2017]    |     61.76     |
|  Single model  |     MBMN(MVM)      |     60.03     |
|                |     MBMN(SMVM)     |   **61.97**   |
|                |   MBMN(MVM)+SMN    |     62.11     |
| Combined model |   MBMN(SMVM)+SMN   |     62.08     |
|                | MBMN(MVM)+SMN+NLP  |   **62.26**   |
|                | MBMN(SMVM)+SMN+NLP |     62.16     |
