from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch
import json
from src.data.components.context_util import Task3ContextUtility
from src.data.components.dataset import SingleArticleLegalDataset
from src.models.lightning_module import SingleArticleModel
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils.bert_utils import evalute_list_sample_recall, evaluate_list_sample_precision, evalute_list_sample_f2

BATCH_SIZE = 1
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

TRAIN_CANDIDATE = 30
VALIDATION_CANDIDATE = 100 
TEST_CANDIDATE = 95 

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

def add_bert_to_sample(
    list_data,
    _output,
    set_prediction= True,
    output_file_path= None
):
    num_empty = 0
    for sample in list_data:
      try:
        q_id = sample['question_id']
        scores = []
        for q in _output:
          if q['list_qid'][0] == q_id and q['label'] == 1:
            article = q['list_article_identity'][0]
            prob = torch.max(torch.nn.functional.softmax(q['model_predict'].detach().cpu().float(), dim= -1)).item()
            scores.append((article, prob))
        articles = {k: v for k, v in scores}
        articles = dict(sorted(articles.items(), key=lambda item: item[1], reverse= True))
        sample['bert_candidate'] = list(articles.keys())
        sample['bert_prob'] = articles
      except Exception as e:
        print(e)
        print(sample['question_id'])
        print(q)
        num_empty += 1
    print(num_empty)
    json.dump(
        list_data, open(output_file_path, "w", encoding="utf-8"), ensure_ascii=False
    ) if output_file_path is not None else None

# Tạo instance logger, làm nhiệm vụ ghi log
# logger = TensorBoardLogger(save_dir='./logs/', version="single-article-BERT", log_graph=True)
logger = WandbLogger(save_dir='./logs', project="legal-document-retrieval")
ckpt_callback = ModelCheckpoint(dirpath= './weights/', 
                                    filename= 'ldr_run3_{epoch:02d}_{val_loss:0.3f}',
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

RESUME_CHECKPOINT = './weights/ldr_epoch=04_val_loss=0.121.ckpt'

model = model_object.load_from_checkpoint(RESUME_CHECKPOINT)

# trainer = Trainer(resume_from_checkpoint="./weights/ldr_epoch=04_val_loss=0.121.ckpt")
# print("Start validating...")
# trainer.validate(model=model, dataloaders=valid_dataloader)

print("Start testing...")
trainer.test(model=model, dataloaders=test_dataloader)

W2V_TEST_PATH = "./data/added_w2v_test_data.json"
OUTPUT_BERT_TEST_PATH = "./data/added_bert_test_data.json"

with open(W2V_TEST_PATH, 'r', encoding='utf-8') as f:
  w2v_test_data = json.load(f)
  
add_bert_to_sample(
    w2v_test_data,
    model.validation_step_outputs,
    output_file_path= OUTPUT_BERT_TEST_PATH
)

print('Recall score: ')
print(evalute_list_sample_recall(w2v_test_data))
print('='*10)
print('Precision score: ')
print(evaluate_list_sample_precision(w2v_test_data))
print('='*10)
print('F2 score: ')
print(evalute_list_sample_f2(w2v_test_data))

print("Finished!")