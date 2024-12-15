# File for computing edit Affinity and Diversity
import json
import os

import scipy
from collections import Counter
from evaluation.augmenters.data import Dataset, M2DataReader


def build_pattern(data: Dataset):
    cnt_edit = Counter()
    for sample in data:
        for src_edits in sample.edits:
            for src_tgt_edits in src_edits:
                for edit in src_tgt_edits:
                    cnt_edit[(" ".join(edit.src_tokens), " ".join(edit.tgt_tokens))] += 1
    return cnt_edit


def calc_edit_affinity(data_ref: Dataset, data_hyp: Dataset):
    """ Compute Edit Distribution Distance between data_ref and data_hyp
        1) Extract two sets of edits
        2) Compute KL Divergence
    """
    patterns_ref = build_pattern(data_ref)
    patterns_hyp = build_pattern(data_hyp)

    dist_ref, dist_hyp = [], []
    for (src_tokens, tgt_tokens), cnt in patterns_ref.items():
        dist_ref.append(cnt)
        dist_hyp.append(patterns_hyp[(src_tokens, tgt_tokens)])
    kl_hyp_ref = scipy.stats.entropy(dist_hyp, dist_ref)

    dist_ref, dist_hyp = [], []
    for (src_tokens, tgt_tokens), cnt in patterns_hyp.items():
        dist_hyp.append(cnt)
        dist_ref.append(patterns_ref[(src_tokens, tgt_tokens)])
    kl_ref_hyp = scipy.stats.entropy(dist_ref, dist_hyp)

    return 2 / (kl_ref_hyp + kl_hyp_ref)


def calc_edit_diversity(data: Dataset):
    """ Compute Edit Distribution Distance between data_ref and data_hyp
        1) Extract the edit set
        2) Compute joint entropy
    """
    patterns = build_pattern(data)
    dist_edit = list(patterns.values())
    return scipy.stats.entropy(dist_edit)

def save_texts_to_files(data, source_file,target_file):
    with open(source_file, 'w') as source_f, open(target_file, 'w') as masked_f:
        for entry in data:
            source_f.write(entry["source_text"] + "\n")
            masked_f.write(entry["target_text"] + "\n")
def calc_ad(hyp_file, ref_file):
    with open(hyp_file, 'r') as f:
        data = json.load(f)
    save_texts_to_files(data, 'data/txt/source_origin.txt', 'data/txt/target_origin.txt')
    os.system("errant_parallel -orig data/txt/source_origin.txt -cor data/txt/target_origin.txt -out data/m2/origin")
    with open(ref_file, 'r') as f:
        data = json.load(f)
    save_texts_to_files(data, 'data/txt/source_aug.txt', 'data/txt/target_aug.txt')
    os.system("errant_parallel -orig data/txt/source_aug.txt -cor data/txt/target_aug.txt -out data/m2/aug")
    reader_m2 = M2DataReader()
    data_hyp = reader_m2.read("data/m2/origin")
    data_ref = reader_m2.read("data/m2/aug")

    # Calculate Affinity
    affinity = calc_edit_affinity(data_ref, data_hyp)
    print(f"Affinity between hyp and ref: {round(affinity, 4)}")

    # Calculate Diversity
    diveristy_ref = calc_edit_diversity(data_ref)
    diveristy_hyp = calc_edit_diversity(data_hyp)
    print(f"Dataset Diversity: of ref: {round(diveristy_ref, 4)}")
    print(f"Dataset Diversity: of hyp: {round(diveristy_hyp, 4)}")