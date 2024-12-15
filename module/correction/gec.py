import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
from module.config import Config
from module.augmentation.mask.dataset import dataset
from module.augmentation.mask.model import T5Model, WarmupScheduler, EarlyStopping
from transformers import T5Tokenizer
config = Config()
tokenizer = T5Tokenizer.from_pretrained("checkpoint/t5-base", legacy=False)
def train(data, dev_data):
    train_loader, dev_loader = dataset(data, dev_data)
    model = T5Model(model_name="checkpoint/t5-base", config=config).to(
        config.device)
    if os.path.exists(config.model_path):
        print(f"Loading model from {config.model_path}")
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup_scheduler = WarmupScheduler(optimizer, total_warmup_steps=config.warmup_steps)
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            optimizer.zero_grad()
            output, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, inference=False)
            loss.backward()
            optimizer.step()
            warmup_scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Total Loss: {total_loss / len(train_loader)}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader):
                input_ids = batch["input_ids"].to(config.device)
                labels = batch["labels"].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                optimizer.zero_grad()
                output, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, inference=False)
                optimizer.step()
                warmup_scheduler.step()
                total_val_loss += loss.item()

        print(f"Validation Loss: {total_val_loss / len(dev_loader)}")
        torch.save(model.state_dict(), config.model_path)
        early_stopping(total_val_loss / len(dev_loader))
        if early_stopping.early_stop:
            print("Early stopping")
            break

def test(data):
    model = T5Model(model_name="checkpoint/t5-base",
                    config=config).to(
        config.device)
    if os.path.exists(config.model_path):
        print(f"Loading model from {config.model_path}")
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model.eval()
    predictions, sources, targets = [], [], []

    for item in tqdm(data):
        input_ids = tokenizer.encode(item, return_tensors="pt").to(config.device)
        attention_mask = torch.ones(input_ids.shape, device=config.device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, inference=True)
        pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(pred_text)


    with open(config.test_out_path, 'w', encoding='utf-8') as f:
        for line in predictions:
            f.write(f"{line}\n")