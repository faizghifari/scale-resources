import json
import csv

def dict_to_text(d):
    if not d:  # If dictionary is empty
        return ""
    return '\n'.join(f"{k} -> {v}" for k, v in d.items())

def json_to_csv(input_file, output_file):
    # Read JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get all keys from the first item to use as CSV headers
    headers = data[0].keys()
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for item in data:
            # Convert dictionary fields to text format
            item['changed_words'] = dict_to_text(item['changed_words'])
            item['ngoko_to_bebasan_changes'] = dict_to_text(item['ngoko_to_bebasan_changes'])
            writer.writerow(item)

# Usage example:
json_to_csv('output_translations_with_bebasan.json', 'output.csv')