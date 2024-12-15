import spacy

class M2parser:
    def __init__(self):
        self.entries = []
        self.noop = []
        self.entry_id = 0
        self.noop_id = 0
        self.nlp = spacy.load("en_core_web_sm")

    def edit(self, source_text, edits):
        words = source_text.split()
        for start, end, correction in sorted(edits, key=lambda x: x[0], reverse=True):
            if correction:
                words[start:end] = [correction]
            else:
                del words[start:end]
        target_text = " ".join(words)
        words = source_text.split()
        mask_count = 0
        total_length = 0
        for start, end, _ in sorted(edits, key=lambda x: x[0], reverse=True):
            if start >= len(words):
                continue
            mask_len = max(end - start, 1)
            words[start:end] = ["<mask>"] * mask_len
            mask_count += mask_len
            total_length += len("<mask>") * mask_len
        masked_text = " ".join(words)
        mask_count = masked_text.count("<mask>")
        total_count = len(masked_text.split())
        mask_rate = mask_count / total_count if total_count else 0
        return target_text, masked_text, round(mask_rate, 4)

    def parse_m2_entry(self, source_line, annotations):
        source_text = source_line[2:].strip()
        edits = []
        error_types = []
        error_positions = []
        for annotation in annotations:
            parts = annotation[2:].split("|||")
            if len(parts) < 2:
                continue
            error_range = parts[0].split()
            error_type = parts[1]
            correction = parts[2] if len(parts) > 2 else ""
            start, end = int(error_range[0]), int(error_range[1])
            edits.append((start, end, correction))
            error_types.append(error_type)
            error_positions.append(f"{start}, {end}")
        target_text, masked_text, mask_rate = self.edit(source_text, edits)
        return {
            "id": self.entry_id,
            "source_text": source_text,
            "target_text": target_text,
            "masked_text": masked_text,
            "error_type": ", ".join(set(error_types)),
            "error_positions": "| ".join(error_positions),
            "mask_rate": mask_rate
        }

    def parse_m2_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        source_line = ""
        annotations = []
        for line in lines:
            if line.startswith("S "):
                if source_line:
                    entry = self.parse_m2_entry(source_line, annotations)
                    if entry['error_type'] != 'noop':
                        self.entries.append(entry)
                        self.entry_id += 1
                    else:
                        self.noop.append(entry)
                        self.noop_id += 1
                source_line = line.strip()
                annotations = []
            elif line.startswith("A "):
                annotations.append(line.strip())
        if source_line:
            entry = self.parse_m2_entry(source_line, annotations)
            if entry['error_type'] != 'noop':
                self.entries.append(entry)
                self.entry_id += 1
            else:
                self.noop.append(entry)
                self.noop_id += 1

        return self.entries, self.noop



