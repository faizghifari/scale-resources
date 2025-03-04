import json


def count_tokens(text):
    if not text or text == "-":
        return 0
    return len(text.split())


def is_sentence_example_valid(example):
    # Check if either field is empty or just contains "-"
    balinese = example.get("Balinese", "")
    indonesian = example.get("Indonesian", "")

    if not balinese or not indonesian or balinese == "-" or indonesian == "-":
        return True

    # Count tokens
    balinese_tokens = count_tokens(balinese)
    indonesian_tokens = count_tokens(indonesian)

    # Check if difference is too large
    if abs(balinese_tokens - indonesian_tokens) > 15:
        return False

    return True


def find_invalid_entries(data):
    invalid_keys = []

    for key, value in data.items():
        sentence_examples = value.get("sentence_examples", [])

        # Skip if no examples
        if not sentence_examples:
            continue

        # Check each example
        for example in sentence_examples:
            if not is_sentence_example_valid(example):
                invalid_keys.append(key)
                break

    return invalid_keys


def clean_dictionary(data):
    for key, value in data.items():
        # Get translations
        eng_translations = value.get("translation_english", [])
        ind_translations = value.get("translation_indonesian", [])

        # Remove exact matches with key
        eng_translations = [t for t in eng_translations if t.lower() != key.lower()]
        ind_translations = [t for t in ind_translations if t.lower() != key.lower()]

        # Update only if there are translations
        if eng_translations:
            value["translation_english"] = eng_translations
        if ind_translations:
            value["translation_indonesian"] = ind_translations


def run_invalid_entries_check():
    # Read JSON file
    with open("./dict/transformed_bali_dict.json", "r") as f:
        dictionary = json.load(f)

    # Find invalid entries
    invalid_keys = find_invalid_entries(dictionary)

    # Write results to file
    with open("invalid_entries.txt", "w") as f:
        f.write("Invalid dictionary entries:\n")
        for key in invalid_keys:
            f.write(f"- {key}\n")

    print(
        f"Found {len(invalid_keys)} invalid entries. Results written to invalid_entries.txt"
    )


def run_clean_dictionary():
    # Read the original dictionary
    with open("./dict/transformed_bali_dict.json", "r") as f:
        dictionary = json.load(f)

    # Clean the dictionary
    clean_dictionary(dictionary)

    # Write the cleaned dictionary back to file
    with open("./dict/transformed_bali_dict_.json", "w") as f:
        json.dump(dictionary, f, indent=2)

    print("Dictionary cleaned successfully.")


if __name__ == "__main__":
    run_invalid_entries_check()
    # run_clean_dictionary()
