import torch
import os
from torch.optim import AdamW
from tqdm import tqdm
from module.config import Config
from module.augmentation.mask.dataset import dataset
from module.augmentation.mask.model import T5Model, WarmupScheduler, EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = Config()
device = config.device
def MaskTrainer(data):
    train_loader, dev_loader = dataset(data)
    model_path = "checkpoint/t5_model_mask.pth"
    model = T5Model(model_name='checkpoint/t5-base', config=config).to(config.device)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    warmup_scheduler = WarmupScheduler(optimizer, total_warmup_steps=config.warmup_steps)
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch['attention_mask'].to(device)
            optimizer.zero_grad()
            output_original, output_noised, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, inference=False)
            loss.backward()
            optimizer.step()
            warmup_scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Total Loss: {total_loss / len(train_loader)}")
        model_save_path = f"checkpoint/t5_model_mask_{epoch}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch['attention_mask'].to(device)
                optimizer.zero_grad()
                output_original, output_noised, loss = model(input_ids=input_ids, attention_mask=attention_mask,
                                                             labels=labels, inference=False)
                optimizer.step()
                warmup_scheduler.step()
                total_val_loss += loss.item()

        print(f"Validation Loss: {total_val_loss / len(dev_loader)}")
        early_stopping(total_val_loss / len(dev_loader))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model.state_dict(), "checkpoint/t5_model_mask.pth")