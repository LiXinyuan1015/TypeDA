from transformers import T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from module.config import Config

config = Config()
tokenizer = T5Tokenizer.from_pretrained('checkpoint/t5-base')
tokenizer.add_tokens(['<mask>'])
def preprocess_text(source_text, target_text):
    prefix = "mask words with <mask>: "
    source_text = prefix + source_text
    return source_text, target_text


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                       padding_value=tokenizer.pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {'input_ids': input_ids_padded, 'attention_mask': attention_mask_padded, 'labels': labels_padded}


class DataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_text = item['source_text']
        target_text = item['masked_text']
        if not source_text or not target_text:
            raise ValueError("Source or target text is empty")
        source_text, target_text = preprocess_text(source_text, target_text)

        source_encodings = tokenizer(source_text, max_length=config.max_length, padding="max_length", truncation=True,
                                     return_tensors="pt")
        target_encodings = tokenizer(target_text, max_length=config.max_length, padding="max_length", truncation=True,
                                     return_tensors="pt")
        return {
            'input_ids': source_encodings['input_ids'].squeeze(),
            'attention_mask': source_encodings['attention_mask'].squeeze(),
            'labels': target_encodings['input_ids'].squeeze()
        }


def dataset(data):
    train_data, val_data = train_test_split(data, test_size=0.1)
    train_dataset = DataSet(train_data)
    val_dataset = DataSet(val_data)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    return train_loader, val_loader

