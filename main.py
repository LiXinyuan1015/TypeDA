import json
from scripts.m2parser import M2parser
from module.augmentation.mask.method import Mask
from module.augmentation.filling.fill import filling
from module.config import Config
from module.correction.gec import train, test
from evaluation.m2scorer.scorer import scorer
from evaluation.metrics import calc_ad
config = Config()
parser = M2parser()
aug_data, noop_data = parser.parse_m2_file('data/m2/ABC.train.gold.bea19.m2')
mask_data = Mask(aug_data)
with open("data/mask_output.json", "w", encoding="utf-8") as f:
    json.dump(mask_data, f, ensure_ascii=False, indent=4)
print("Masked data has been saved to data/mask_output.json")
filling_data = filling("data/mask_output.json", "data/filling_output.json")
train_data = filling_data + noop_data
dev_data = parser.parse_m2_file('data/m2/ABCN.dev.gold.bea19.m2')
train(train_data, dev_data)
with open(config.test_path, 'r', encoding='utf-8') as file:
    test_data = [line.strip() for line in file]
test(test_data)
scorer(config.test_out_path, "data/m2/conll14.gold")
calc_ad("data/filling_output.json", "data/reference.json")

