from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer
import torch
import json
from typing import Dict
from src.data.components.context_util import Task3ContextUtility
from src.data.components.dataset import SingleArticleLegalDataset
from src.models.lightning_module import SingleArticleModel
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data.components.preprocessing import preprocess_data
from rank_bm25 import BM25Okapi
from src.models.bm25 import add_bm25_to_sample
from sklearn.model_selection import train_test_split


BATCH_SIZE = 4
N_EPOCH = 30
NUM_WORKERS = 2
LIMIT_VAL_BATCH = 1
FAST_DEV_RUN = False
PRECISION = 16
if torch.cuda.is_available():
    ACCELERATOR = 'gpu'
    devices = [0]
else:
    devices = 1
    ACCELERATOR = 'cpu'

TRAIN_CANDIDATE = 30
VALIDATION_CANDIDATE = 100 
TEST_CANDIDATE = 95 

processed_data_available = False
if processed_data_available is False:
    # Preprocess data
    law_corpus, stage_2_data, stage_2_test_data = preprocess_data(dump_data=True)

    # Get BM25 added data
    data_text_field = "text"

    corpus_text_field = "text"

    tokenized_corpus = [
        corpus.get(corpus_text_field).split(" ") for corpus in law_corpus
    ]

    bm25_model = BM25Okapi(tokenized_corpus)

    TOP_N = 100

    OUTPUT_FILE_PATH = "./data/added_BM25_data.json"
    train_data = add_bm25_to_sample(stage_2_data, OUTPUT_FILE_PATH, law_corpus, bm25_model, top_n=TOP_N, dump_data=True)

    TEST_OUTPUT_FILE_PATH = "./data/added_BM25_test.json"
    tmp_test_data = add_bm25_to_sample(stage_2_test_data, TEST_OUTPUT_FILE_PATH, law_corpus, bm25_model, top_n=TOP_N, dump_data=True)

    # Split original test data into validation and test data
    VALID_PATH = "./data/added_BM25_valid_data.json"

    TEST_PATH = "./data/added_BM25_test_data.json"

    valid_data, test_data = train_test_split(tmp_test_data, test_size=0.5, shuffle=True, random_state=42)

    with open(VALID_PATH, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False)

    with open(TEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
else:
    LAW_CORPUS_PATH = "./data/stage_2_law_corpus.json"
    TRAIN_PATH = "./data/added_BM25_data.json"
    VALID_PATH = "./data/added_BM25_valid_data.json"
    TEST_PATH = "./data/added_BM25_test_data.json"

    with open(LAW_CORPUS_PATH, 'r', encoding='utf-8') as f:
        law_corpus = json.load(f)

    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(VALID_PATH, 'r', encoding='utf-8') as f:
        valid_data = json.load(f)

    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)


################################################################
context_utility = Task3ContextUtility(statute_corpus=law_corpus)

train_gen_sample = context_utility.create_gen_sample(
    train_data, is_include_gold=True, n_candidate=TRAIN_CANDIDATE
)
valid_gen_sample = context_utility.create_gen_sample(
    valid_data, is_include_gold=True, n_candidate=VALIDATION_CANDIDATE
)
test_gen_sample = context_utility.create_gen_sample(
    test_data, is_include_gold=True, n_candidate=TEST_CANDIDATE
)

dataset_object = SingleArticleLegalDataset
collate_fn = SingleArticleModel.custom_collate_fn
model_object = SingleArticleModel

train_dataset = dataset_object(train_gen_sample)
valid_dataset = dataset_object(valid_gen_sample)
test_dataset = dataset_object(test_gen_sample)

train_dataloader = DataLoader(
    dataset=train_dataset,
    collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
valid_dataloader = DataLoader(
    dataset=valid_dataset,
    collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# Tạo instance logger, làm nhiệm vụ ghi log
# logger = TensorBoardLogger(save_dir='./logs/', version="single-article-BERT", log_graph=True)
logger = WandbLogger(save_dir='./logs', project="legal-document-retrieval")
ckpt_callback = ModelCheckpoint(dirpath= './weights/', 
                                    filename= 'ldr_{epoch:02d}_{val_loss:0.3f}',
                                    monitor= 'val_loss', 
                                    save_on_train_epoch_end= False,
                                    save_top_k= 1,
                                    mode="min"
                                )

# Tạo instance Trainer dựa vào các tham số đã định nghĩa
trainer = Trainer(
        accelerator=ACCELERATOR,
        max_epochs=N_EPOCH,
        default_root_dir='./',
        devices=devices,
        log_every_n_steps=2,
        logger=logger,
        # limit_val_batches=LIMIT_VAL_BATCH,
        fast_dev_run=FAST_DEV_RUN,
        precision=PRECISION,
        callbacks=ckpt_callback
    )

model = model_object(lr=5e-6, eps=1e-8)

print("Start training...")
trainer.fit(
  model=model,
  train_dataloaders=train_dataloader,
  val_dataloaders=valid_dataloader,
)

print("Training finished!")
