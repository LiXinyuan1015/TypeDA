import torch
import torch.nn.functional as F
from transformers import T5Tokenizer
from tqdm import tqdm
import os
from module.config import Config
from module.augmentation.mask.model import T5Model
config = Config()
tokenizer = T5Tokenizer.from_pretrained("checkpoint/t5-base")
tokenizer.add_tokens(['<mask>'])
device = config.device

def add_mask(sequences, scores):
    threshold = config.threshold
    mask_id = tokenizer.convert_tokens_to_ids('<mask>')
    masked_sequences = []
    for sequence, logits_list in zip(sequences, scores):
        sequence_ids = sequence.tolist()
        for i, logits in enumerate(logits_list):
            probs = F.softmax(logits, dim=-1)
            max_probs, best_tokens = probs.max(dim=-1)
            if max_probs < threshold:
                sequence_ids[i] = mask_id
        masked_sequences.append(torch.tensor(sequence_ids))
    return masked_sequences

def inference(data, num, model_path):
    model = T5Model(model_name="checkpoint/t5-base", config=config)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    result = []
    for item in tqdm(data, desc="Processing"):
        input_text = item["source_text"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape, device=device)
        output, output_noised, logits, logits_noised = model(input_ids=input_ids, attention_mask=attention_mask,
                                                             labels=None, inference=True)
        output = add_mask(output, logits)
        output_noised = add_mask(output_noised, logits_noised)
        original_text = tokenizer.decode(output[0], skip_special_tokens=True)
        noised_text = tokenizer.decode(output_noised[0], skip_special_tokens=True)

        for text_content in [item["masked_text"], original_text, noised_text]:
            record = {
                "num": num,
                "id": item["id"],
                "source_text": item["source_text"],
                "target_text": item["target_text"],
                "masked_text": text_content
            }
            result.append(record)
            num += 1
    return result, num


