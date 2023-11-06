from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import torch
import json
from typing import Dict
from src.data.components.context_util import Task3ContextUtility
from src.data.components.dataset import SingleArticleLegalDataset
from src.models.lightning_module import SingleArticleModel

BATCH_SIZE = 4
N_EPOCH = 10
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

LAW_CORPUS_PATH = "./data/stage_2_law_corpus.json"
TRAIN_PATH = "./data/added_BM25_data.json"
VALID_PATH = "./data/added_BM25_valid_data.json"
TEST_PATH = "./data/added_BM25_test_data.json"
TRAIN_CANDIDATE = 30
VALIDATION_CANDIDATE = 100 
TEST_CANDIDATE = 95 

with open(LAW_CORPUS_PATH, 'r', encoding='utf-8') as f:
    law_corpus = json.load(f)

with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open(VALID_PATH, 'r', encoding='utf-8') as f:
    valid_data = json.load(f)

with open(TEST_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

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
logger = TensorBoardLogger(save_dir='./logs/', version="single-article-BERT", log_graph=True)

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
    )

model = model_object()

trainer.fit(
  model=model,
  train_dataloaders=train_dataloader,
  val_dataloaders=valid_dataloader,
)

