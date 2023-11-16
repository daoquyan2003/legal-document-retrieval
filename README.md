# LEGAL DOCUMENT RETRIEVAL TASK ON ALQAC 2023 DATASET

## Data

The dataset we use includes ALQAC 2023, ALQAC 2022 and Zalo legal datasets.

The public test set is not used since it does not contain labels for evaluation.

We split the dataset into three parts: the training set, the validation set and the test set with the ratio of 0.8, 0.1 and 0.1 respectively.

## Model

We traditional models such as BM25, TF-IDF and Word2Vec (combined with k-NearestNeighbors) to retrieve the most relevant articles of each question (query).

We also utilize the output of BM25 to fine-tune the mBERT classification model. mBERT with its contextual embeddings helps extract more concise and accurate law articles.

Finally, we ensemble the outputs of the 4 models above using Borda count (Tournament variant) voting strategy to get the final result.

## How to run the code

First, run `python -m src.data.components.preprocessing` to flatten the data and apply word segmentation.

After that, run `python -m src.models.bm25`, `python -m src.models.bm25_test`, `python -m src.models.tfidf_exec` and `python -m src.models.word2vec` consecutively to get the data with added results from the 3 models.

Next, run `python -m src.data.components.split_data` to get the validation set and test set for mBERT.

Run `python -m src.train` to finetune mBERT model with the processed data.

After getting the weight of finetuned mBERT, change the weight path in `src/test.py` and run `python -m src.test` to evaluate the model on the test set.

Finally, `python -m src.voting` to ensemble the output to get the result.
