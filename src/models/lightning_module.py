import torch
from pytorch_lightning.core.module import LightningModule
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
from src.utils.bert_utils import synthetic_step_output_v2, save_csv_predict_result

class SingleArticleModel(LightningModule):
    def __init__(self, pretrained_model = "bert-base-multilingual-cased"):
        super().__init__()
        self.bert_encoder = BertForSequenceClassification.from_pretrained(pretrained_model).to(self.device)
        self.validation_step_outputs = []
        self.label_weight = None

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, **kwargs) -> Tensor:
        key_pair_input_ids = input_ids[:, 0, :]
        key_pair_attention_mask = attention_mask[:, 0, :]
        key_pair_token_type_ids = token_type_ids[:, 0, :]

        sco = self.bert_encoder(input_ids=key_pair_input_ids, attention_mask=key_pair_attention_mask,
                                token_type_ids=key_pair_token_type_ids)
        return sco.logits

    # Định nghĩa việc cần làm cho mỗi training step
    def training_step(self, sample, sample_idx):
        """
        Do số lượng nhãn dương và âm khá chênh lệch nên trong quá trình huấn luyện
        cần đưa vào thêm một yếu tố đó là trọng số cho mỗi nhãn
        """
        if self.label_weight is None:
            self.label_weight = torch.tensor([1, 30], dtype=torch.float).to(self.device)
        model_input, list_qid, list_aid, bm25_score, label = sample
        model_predict = self.forward(**model_input, label=label)
        label = label.long().to(self.device)
        loss = cross_entropy(input=model_predict, target=label, weight=self.label_weight)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    # Định nghĩa việc cần làm cho mỗi validation step
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        model_input, list_qid, list_aid, bm25_score, label = batch
        model_predict = self.forward(**model_input)
        label = label.to(torch.device('cpu'))
        self.validation_step_outputs.append(
            {'list_qid': list_qid,
             'model_predict': model_predict,
             'label': label,
             'list_article_identity': list_aid,
             'bm25_score': bm25_score}
            )

    # Định nghĩa việc cần làm sau mỗi epoch validation
    def on_validation_epoch_end(self) -> None:
        save_qid_list, \
            save_article_identity_list, \
            save_label_list, \
            save_bm25_predict_list, \
            save_bert_predict_list = synthetic_step_output_v2(self.validation_step_outputs)

        save_csv_predict_result(save_qid_list, save_article_identity_list,
                                save_label_list, save_bm25_predict_list, save_bert_predict_list)

    # Định nghĩa việc cần làm tại mỗi test step
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.validation_step(batch, batch_idx)

    # Định nghĩa việc cần làm sau mỗi test epoch
    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        return torch.optim.AdamW(params=self.parameters(), lr=2e-5, eps=1e-8)

    # Định nghĩa hàm xử lý trong Dataloader, trước khi đưa vào mô hình
    @staticmethod
    def custom_collate_fn(batch):
        list_qid = [e[1] for e in batch]
        list_article_id = [e[2] for e in batch]
        list_key = batch[0][0].keys()
        model_input = {key: pad_sequence([e[0][key] for e in batch], batch_first=True) for key in list_key}
        bm25_score = torch.tensor([e[3] for e in batch])
        output = torch.tensor([e[4] for e in batch])

        # outputs = torch.as_tensor([e[1] for e in batch])
        return model_input, list_qid, list_article_id, bm25_score, output