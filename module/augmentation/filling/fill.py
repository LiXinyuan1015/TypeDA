import os
import json
from openai import OpenAI
from tqdm import tqdm
from module.config import Config
from module.augmentation.filling.tool import sampling
from module.augmentation.filling.template import construction
from module.augmentation.filling.postprocess import postprocess
config = Config()
def api_call(masked_item, error_type, model, client):
    input = construction(masked_item, error_type)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": input}]
    )
    output = response.choices[0].message.content
    return output

def filling(input_path, output_path):
    model = "gpt-4"
    api = config.api_key
    client = OpenAI(
        base_url='https://api.openai.com/v1',
        api_key=api,
    )
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as file:
            filling_data = json.load(file)
    else:
        filling_data = []
    processed_lines = len(filling_data)
    with open(input_path, "r", encoding='utf-8') as file:
        data = json.load(file)
    for item in tqdm(data[processed_lines:]):
        error_type = sampling()[0]
        output = api_call(item, error_type, model, client)
        i = {
            "id": item['num'],
            "source_text": output,
            "target_text": item["target_text"],
            "error_type": error_type
        }
        filling_data.append(i)
        filling_data = postprocess(filling_data)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filling_data, f, ensure_ascii=False, indent=4)
        print("Filling data has been saved to data/filling_output.json")
        return filling_data

