from torch.utils.data import Dataset
from transformers import BertTokenizer

class SingleArticleLegalDataset(Dataset):
    def __init__(self, list_gen_sample, pretrained_model = "bert-base-multilingual-cased"):
        self.list_gen_sample = list_gen_sample
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def __len__(self):
        return len(self.list_gen_sample)

    def __getitem__(self, idx):
        gen_sample = self.list_gen_sample[idx]
        tokenized_pair = self.tokenizer(text=gen_sample.get('q'), text_pair=gen_sample.get('a'), padding='max_length',
                                        return_tensors='pt', truncation='longest_first', return_token_type_ids=True)
        return tokenized_pair, gen_sample.get('qid'), gen_sample.get('aid'), \
            gen_sample.get('bm25_score'), gen_sample.get('label')
