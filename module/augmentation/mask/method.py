import numpy as np
from module.augmentation.mask.trainer import MaskTrainer
from module.augmentation.mask.infer import inference
from module.config import Config
config = Config()

import spacy
from module.config import Config
config = Config()
def find_closest_mask_index(masked_words, source_word, source_index):
    range_start = max(0, source_index - 3)
    range_end = min(len(masked_words), source_index + 4)
    if source_word in masked_words[range_start:range_end]:
        return None
    closest_index = None
    closest_distance = float('inf')

    for i, word in enumerate(masked_words):
        if word == "<mask>":
            distance = abs(i - source_index)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

    return closest_index

def SRL(data):
    for entry in data:
        if entry["mask_rate"] >= config.mask_rate:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(entry["source_text"])
            source_words = [token.text for token in doc]
            subject_index = next((i for i, token in enumerate(doc) if token.dep_ == "nsubj"), None)
            verb_index = next((i for i, token in enumerate(doc) if token.dep_ == "ROOT"), None)

            masked_words = entry["masked_text"].split()
            subj_mask_index = find_closest_mask_index(masked_words, source_words[subject_index],
                                                      subject_index) if subject_index is not None else None
            verb_mask_index = find_closest_mask_index(masked_words, source_words[verb_index],
                                                      verb_index) if verb_index is not None else None
            if subj_mask_index is not None:
                masked_words[subj_mask_index] = source_words[subject_index]
            if verb_mask_index is not None:
                masked_words[verb_mask_index] = source_words[verb_index]
            entry["masked_text"] = ' '.join(masked_words)
            mask_count = entry["masked_text"].count("<mask>")
            total_count = len(entry["masked_text"].split())
            entry["mask_rate"] = mask_count / total_count if total_count else 0
    return data


def filter(data):
    filtered_data = []
    num = 0
    data = [item[0] for item in data][0]
    for item in data:
        masked_text = item["masked_text"]
        words = masked_text.split()
        mask_count = words.count("<mask>")
        total_words = len(words)
        if mask_count > 0 and mask_count <= total_words * 0.5:
            item['num'] = num
            filtered_data.append(item)
            num += 1
    return filtered_data
def Mask(data):
    data = SRL(data)
    k_fold = config.k_fold
    data = np.array_split(data, k_fold)
    results = []
    num = 0
    for i in range(0, k_fold):
        train_data = np.concatenate([data[j] for j in range(k_fold) if j != i], axis=0)
        val_data = data[i]
        MaskTrainer(train_data)
        model_path = "checkpoint/t5_model_mask.pth"
        result = inference(val_data, num, model_path)
        results.append(result)
    masked_data = filter(results)
    return masked_data