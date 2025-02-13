import json

def combine_and_clean_json(file1_path, file2_path, output_file_path):
    # Load the JSON files
    with open(file1_path, 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    with open(file2_path, 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)

    # Combine the JSON data
    combined_data = {**data1, **data2}

    # Clean the combined data
    cleaned_data = {}
    for key, values in combined_data.items():
        if key.strip():
            cleaned_key = key.strip().lower()
            cleaned_values = list(set(value.strip().lower() for value in values if value.strip()))
            if cleaned_values:
                cleaned_data[cleaned_key] = cleaned_values

    # Save the cleaned combined data to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(cleaned_data, output_file, ensure_ascii=False, indent=4)

    print(f"Combined and cleaned JSON data has been saved to '{output_file_path}'")

# Example usage
combine_and_clean_json('./dict/cbn_idn1.json', './dict/cbn_idn2.json', './dict/cbn_idn.json')
combine_and_clean_json('./dict/idn_cbn1.json', './dict/idn_cbn2.json', './dict/idn_cbn.json')