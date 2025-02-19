import json


def transform_dictionary(input_file, output_file):
    # Read the input JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Create new dictionary
    new_dict = {}

    # Process each letter group
    for letter_group in data.values():
        # Process each word entry
        for entry in letter_group:
            word = entry["word"].lower()

            # Process English translations
            eng_trans = entry["translation_english"]
            if eng_trans and eng_trans not in ["-", "—"]:
                eng_list = [t.strip().lower() for t in eng_trans.split(";")]
                eng_list = [t for t in eng_list if t and t != "-" and t != "—"]
            else:
                eng_list = []

            # Process Indonesian translations
            indo_trans = entry["translation_indonesian"]
            if indo_trans and indo_trans not in ["-", "—"]:
                indo_list = [t.strip().lower() for t in indo_trans.split(";")]
                indo_list = [t for t in indo_list if t and t != "-" and t != "—"]
            else:
                indo_list = []

            # Process sentence examples
            sentence_examples = []
            for example in entry["sentence_examples"]:
                # Check if all fields are empty or contain only "-" or "—"
                if (
                    example.get("Balinese", "") in ["", "-", "—"]
                    and example.get("English", "") in ["", "-", "—"]
                    and example.get("Indonesian", "") in ["", "-", "—"]
                ):
                    continue
                sentence_examples.append(example)

            # Only add entry if it has at least one translation or sentence example
            if eng_list or indo_list or sentence_examples:
                new_dict[word] = {
                    "translation_english": eng_list,
                    "translation_indonesian": indo_list,
                    "sentence_examples": sentence_examples,
                }

    # Write the transformed dictionary to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=2)


# Execute the transformation
transform_dictionary("dict/bali_dict.json", "dict/transformed_bali_dict.json")
