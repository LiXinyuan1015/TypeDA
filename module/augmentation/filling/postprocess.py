from Levenshtein import distance as levenshtein_distance
import errant
import spacy
from module.config import Config
config = Config()
def postprocess(data):
    filtered_data = []
    threshold = config.distance
    new_num = 0
    sources = []
    targets = []
    results = []
    for item in data:
        source_text = item['source_text']
        target_text = item['target_text']
        edit_distance = levenshtein_distance(source_text, target_text)
        if edit_distance <= threshold and '<mask>' not in source_text:
            item['num'] = new_num
            filtered_data.append(item)
            new_num += 1

    for item in filtered_data:
        source_text = item["source_text"]
        target_text = item["target_text"]
        sources.append(source_text)
        targets.append(target_text)

    nlp = spacy.load("en_core_web_sm")
    annotator = errant.load('en')
    id = 0
    for sor, orig in zip(sources, targets):
        sor_ann = nlp(sor)
        orig_ann = nlp(orig)
        edit_ann = annotator.annotate(sor_ann, orig_ann)
        error = []
        for e in edit_ann:
            error.append(e.type)
        entry = {
            "id": id,
            "source_text": sor,
            "target_text": orig,
            "error_type": ', '.join(error)
        }
        results.append(entry)
        id += 1
    return results