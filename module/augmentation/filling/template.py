import os
import json
from module.augmentation.filling.tool import error_mapping
from module import Config
config = Config()

def examples(error_type):
    example_path = os.path.join(os.path.dirname(os.getcwd()), "data/LLMs Prompting", "example.json")
    description = error_mapping.get(error_type, "Error description not found")
    with open(example_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    result_string = ""
    num = 1
    for item in data:
        if item["error_type"] == error_type:
            result_string += f"Example reference{num}: \n"
            result_string += f"example input: {item['masked_text']}\n"
            result_string += f"wanted error type: {description}\n"
            result_string += f"example filling with wanted error:{item['masking_fill']}\n"
            result_string += f"explanation: {item['explanation']}\n\n"
            num += 1

    return result_string


def Input(item, error_type):
    description = error_mapping.get(error_type, "Error description not found")
    result_string = ""
    result_string += f"original sentence with syntax error: {item['source_text']}\n"
    result_string += f"the syntactically correct version of the original sentence: {item['target_text']}\n"
    result_string += f"masked input:{item['masked_text']}\n"
    result_string += f"please fill the masked input with {description}\n\n"
    return result_string


def construction(masked_item, error_type):
    prefix = "Now I'm going to give you a sentence with several <mask>. I need you to make this sentence with specific grammatical errors by filling in or deleting the <mask>. You must fill in or delete all the <mask>, and other words must be retained and not changed.\nI will provide you with reference examples, the original sentence with the syntax error, the masked sentence, and the syntactically correct version of the original sentence, and give you a certain syntax error that you need to fill the masked sentence into a sentence with the syntax errors I specify. Please make sure that all the <mask> should be filled or deleted and <mask> is not allowed in the output.\n\n"
    example = examples(error_type)
    input = Input(masked_item, error_type)
    output = "The output contains only the filling statements with syntax errors. Do never output any other content."
    template = prefix + example + input + output
    return template
