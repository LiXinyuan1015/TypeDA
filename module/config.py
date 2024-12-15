import torch
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("checkpoint/t5-base", legacy=False)
class Config:
    def __init__(self):
        self.embedding_size = 256
        self.hidden_size = 512
        self.num_layers = 4
        self.weight_decay = 1e-6
        self.dropout = 0.4
        self.input_size = 40000
        self.output_size = 40000
        self.batch_size = 4
        self.num_epochs = 10
        self.mask_rate = 0.5
        self.gaussian_noise_std = 0.5
        self.learning_rate = 5e-5
        self.warmup_steps = 500
        self.patience = 3
        self.k_fold = 5
        self.total_steps = 100000
        self.num_heads = 8
        self.threshold = 0.5
        self.max_length = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "")
        self.alpha = 1.1
        self.lamb = 1.0
        self.api_key = ""
        self.model_path = "checkpoint/gec_model"
        self.test_path = 'data/txt/conll14.txt'
        self.test_out_path = "data/bart_conll14.txt"
        self.distance = 25
        self.weight = 5.0
        self.padding_idx = 0
        self.num_beams = 5
        self.temperature = 1.1
        self.src_pad_idx = tokenizer.pad_token_id
        self.trg_pad_idx = tokenizer.pad_token_id