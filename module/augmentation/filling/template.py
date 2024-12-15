import json
error_mapping = {
    "M:ADJ": "Missing adjective",
    "R:ADJ": "Wrong adjective",
    "U:ADJ": "Unnecessary adjective",
    "R:ADJ:FORM": "Incorrect adjective form",
    "M:ADV": "Missing adverb",
    "R:ADV": "Wrong adverb",
    "U:ADV": "Unnecessary adverb",
    "M:CONJ": "Missing conjunction",
    "R:CONJ": "Wrong conjunction",
    "U:CONJ": "Unnecessary conjunction",
    "M:CONTR": "Missing contraction",
    "R:CONTR": "Wrong contraction",
    "U:CONTR": "Unnecessary contraction",
    "M:DET": "Missing determiner",
    "R:DET": "Wrong determiner",
    "U:DET": "Unnecessary determiner",
    "R:MORPH": "Incorrect morphological form",
    "M:NOUN": "Missing noun",
    "R:NOUN": "Wrong noun",
    "U:NOUN": "Unnecessary noun",
    "R:NOUN:INFL": "Incorrect noun inflection",
    "R:NOUN:NUM": "Incorrect noun number",
    "M:NOUN:POSS": "Missing noun possessive",
    "R:NOUN:POSS": "Incorrect noun possessive",
    "U:NOUN:POSS": "Unnecessary noun possessive",
    "R:ORTH": "Wrong correct spelling",
    "M:PART": "Missing particle",
    "R:PART": "Wrong particle",
    "U:PART": "Unnecessary particle",
    "M:PREP": "Missing preposition",
    "R:PREP": "Wrong preposition",
    "U:PREP": "Unnecessary preposition",
    "M:PRON": "Missing pronoun",
    "R:PRON": "Wrong pronoun",
    "U:PRON": "Unnecessary pronoun",
    "M:PUNCT": "Missing punctuation",
    "R:PUNCT": "Wrong punctuation",
    "U:PUNCT": "Unnecessary punctuation",
    "R:SPELL": "Incorrect spelling",
    "M:VERB": "Missing verb",
    "R:VERB": "Wrong verb",
    "U:VERB": "Unnecessary verb",
    "M:VERB:FORM": "Missing verb form",
    "R:VERB:FORM": "Incorrect verb form",
    "U:VERB:FORM": "Unnecessary verb form",
    "R:VERB:INFL": "Incorrect verb inflection",
    "R:VERB:SVA": "Incorrect subject-verb agreement",
    "M:VERB:TENSE": "Missing verb tense",
    "R:VERB:TENSE": "Incorrect verb tense",
    "U:VERB:TENSE": "Unnecessary verb tense",
    "R:WO": "Wrong word order",
}


def examples(error_type):
    description = error_mapping.get(error_type, "Error description not found")
    with open("data/example.json", 'r', encoding='utf-8') as file:
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
