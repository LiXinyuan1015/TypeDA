import torch
import torch.nn as nn
import torch.nn.functional as F
from module.config import Config
from transformers import T5ForConditionalGeneration, T5Tokenizer
config = Config()

class T5Model(nn.Module):
    def __init__(self, model_name='checkpoint/t5-base', config=config):
        super(T5Model, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['<mask>'])
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.t5_model.resize_token_embeddings(len(self.tokenizer))
        self.config = config
        self.gaussian_noise_std = config.gaussian_noise_std
        self.custom_loss = Loss(pad_idx=self.tokenizer.pad_token_id, mask_idx=self.tokenizer.convert_tokens_to_ids('<mask>'),vocab_size=len(self.tokenizer),
                            weight_for_mask=config.weight, lamb=config.lamb, alpha=config.alpha)

    def forward(self, input_ids, attention_mask, labels, inference):
        encoder_last_hidden_state = self.t5_model.encoder(input_ids=input_ids,
                                                          attention_mask=attention_mask).last_hidden_state
        noise = torch.randn(encoder_last_hidden_state.size()).to(encoder_last_hidden_state.device) * self.gaussian_noise_std
        encoder_last_hidden_state_noised = encoder_last_hidden_state + noise
        if inference:
            output_original = self.t5_model.generate(input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     encoder_outputs=(encoder_last_hidden_state,),
                                                     max_length = config.max_length,
                                                     return_dict_in_generate=True,
                                                     output_scores=True)
            output_noised = self.t5_model.generate(input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   encoder_outputs=(encoder_last_hidden_state_noised,),
                                                   max_length = config.max_length,
                                                   return_dict_in_generate=True,
                                                   output_scores=True)
            original = output_original.sequences
            noised = output_noised.sequences
            logits = output_original.scores
            logits_noised = output_noised.scores
            return original, noised, logits, logits_noised
        output_original = self.t5_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels,
                                        encoder_outputs=(encoder_last_hidden_state,))

        output_noised = self.t5_model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      labels=labels,
                                      encoder_outputs=(encoder_last_hidden_state_noised,))
        loss = self.custom_loss(output_original.logits, output_noised.logits, labels)


        return output_original, output_noised, loss

class WarmupScheduler:
    def __init__(self, optimizer, total_warmup_steps):
        self.optimizer = optimizer
        self.total_warmup_steps = total_warmup_steps
        self.current_step = 0
    def step(self):
        self.current_step += 1
        if self.current_step <= self.total_warmup_steps:
            lr = config.learning_rate * (self.current_step / self.total_warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=config.patience, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


class Loss(nn.Module):
    def __init__(self, pad_idx, mask_idx, vocab_size, weight_for_mask=config.weight, lamb=config.lamb, alpha=config.alpha):
        super(Loss, self).__init__()
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.lamb = lamb
        self.alpha = alpha
        weight = torch.ones(vocab_size)
        weight[mask_idx] = weight_for_mask
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=pad_idx, reduction='sum')
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, output_original, output_noised, target):
        log_probs_original = F.log_softmax(output_original, dim=-1)
        log_probs_noised = F.log_softmax(output_noised, dim=-1)
        target = target.view(-1)
        log_probs_original = log_probs_original.view(-1, log_probs_original.size(-1))
        log_probs_noised = log_probs_noised.view(-1, log_probs_noised.size(-1))
        nll_loss_original = self.nll_loss(log_probs_original, target)
        nll_loss_noised = self.nll_loss(log_probs_noised, target)
        nll_loss = nll_loss_original + nll_loss_noised
        diversity_loss = (log_probs_original - log_probs_noised).abs().mean()
        total_loss = self.lamb * nll_loss + self.alpha * diversity_loss

        return total_loss

