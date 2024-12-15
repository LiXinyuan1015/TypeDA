import json
import random
import math
from collections import Counter
def error_counter(file_path, error_types):
    with open(file_path, 'r') as file:
        data = json.load(file)
    all_error_types = [item["error_type"] for item in data if "error_type" in item]
    error_counts = Counter(all_error_types)
    counts = {error_type: error_counts.get(error_type, 0) for error_type in error_types}
    return counts

def sampling():
    error_types = [
        "M:ADJ", "R:ADJ", "U:ADJ", "R:ADJ:FORM", "M:ADV", "R:ADV", "U:ADV",
        "M:CONJ", "R:CONJ", "U:CONJ", "M:CONTR", "R:CONTR", "U:CONTR", "M:DET",
        "R:DET", "U:DET", "R:MORPH", "M:NOUN", "R:NOUN", "U:NOUN", "R:NOUN:INFL",
        "R:NOUN:NUM", "M:NOUN:POSS", "R:NOUN:POSS", "U:NOUN:POSS", "R:ORTH", "M:PART",
        "R:PART", "U:PART", "M:PREP", "R:PREP", "U:PREP", "M:PUNCT", "R:PUNCT", "U:PUNCT",
        "R:SPELL", "M:VERB", "R:VERB", "U:VERB", "M:VERB:FORM", "R:VERB:FORM", "U:VERB:FORM",
        "R:VERB:INFL", "R:VERB:SVA", "M:VERB:TENSE", "R:VERB:TENSE", "U:VERB:TENSE", "R:WO"
    ]

    error_counts = error_counter("data/reference.json", error_types)
    n_values = [error_counts[error_type] for error_type in error_types]
    probabilities = [n / sum(n_values) for n in n_values]
    adjusted_probabilities = [math.sqrt(p) for p in probabilities]
    total_adjusted = sum(adjusted_probabilities)
    normalized_probabilities = [p / total_adjusted for p in adjusted_probabilities]
    tolerance = 1e-6
    assert abs(sum(normalized_probabilities) - 1) < tolerance, "Total sum of probabilities is not close to 1."
    sampled_error_types = random.choices(error_types, k=1)

    return sampled_error_types


