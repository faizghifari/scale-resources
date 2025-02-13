import os
import re
import json
from collections import defaultdict

# Function to clean words by removing parentheses and their content
def clean_word(word):
    return re.sub(r'\(.*?\)', '', word).strip()

def clean_list(lst):
    return [x for x in lst if x != "" and x != "?" and x != "..."]

def clean_nested_dict(data):
    cleaned_data = {}
    for outer_key, inner_dict in data.items():
        cleaned_data[outer_key] = {}
        for inner_key, value_list in inner_dict.items():
            # Convert list to set to remove duplicates, then back to list
            cleaned_data[outer_key][inner_key] = list(dict.fromkeys(value_list))
    return cleaned_data

def split_comma_slash_string(text):
    # First split by comma, then split each part by slash
    result = []
    for part in text.split(','):
        result.extend(part.split('/'))
    return result

# Function to process each file and extract the dictionaries
def process_file(file_path):
    # Initialize dictionaries for translations
    ngoko_to_bebasan = defaultdict(list)
    bebasan_to_ngoko = defaultdict(list)
    bebasan_to_indonesia = defaultdict(list)
    indonesia_to_bebasan = defaultdict(list)
    ngoko_to_indonesia = defaultdict(list)
    indonesia_to_ngoko = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Skip the first 3 lines
        for _ in range(3):
            next(file)
        
        for line in file:
            # Clean the line (remove extra spaces and split by '|')
            parts = [part.strip() for part in line.split('|') if part.strip()]
            
            if len(parts) < 3:
                continue  # Skip malformed lines
            
            # Clean the words by removing parentheses content
            ngoko_word = clean_word(parts[0])
            bebasan_word = clean_word(parts[1])
            indonesia_word = clean_word(parts[2])
            
            # Handle cases where there are multiple translations separated by commas
            ngoko_words = list(set([word.strip().lower() for word in split_comma_slash_string(ngoko_word)]))
            bebasan_words = list(set([word.strip().lower() for word in split_comma_slash_string(bebasan_word)]))
            indonesia_words = list(set([word.strip().lower() for word in split_comma_slash_string(indonesia_word)]))

            ngoko_words = clean_list(ngoko_words)
            bebasan_words = clean_list(bebasan_words)
            indonesia_words = clean_list(indonesia_words)
            
            # Add translations to dictionaries
            for ngoko in ngoko_words:
                for bebasan in bebasan_words:
                    ngoko_to_bebasan[ngoko].append(bebasan)
                    bebasan_to_ngoko[bebasan].append(ngoko)
                for indonesia in indonesia_words:
                    ngoko_to_indonesia[ngoko].append(indonesia)
                    indonesia_to_ngoko[indonesia].append(ngoko)

            for bebasan in bebasan_words:
                for indonesia in indonesia_words:
                    bebasan_to_indonesia[bebasan].append(indonesia)
                    indonesia_to_bebasan[indonesia].append(bebasan)

    return ngoko_to_bebasan, bebasan_to_ngoko, bebasan_to_indonesia, indonesia_to_bebasan, ngoko_to_indonesia, indonesia_to_ngoko

# Function to process all txt files in a folder
def process_files_in_folder(folder_path):
    # Initialize final dictionaries to store all translations from all files
    ngoko_to_bebasan_all = defaultdict(list)
    bebasan_to_ngoko_all = defaultdict(list)
    bebasan_to_indonesia_all = defaultdict(list)
    indonesia_to_bebasan_all = defaultdict(list)
    ngoko_to_indonesia_all = defaultdict(list)
    indonesia_to_ngoko_all = defaultdict(list)

    # Process each txt file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            ngoko_to_bebasan, bebasan_to_ngoko, bebasan_to_indonesia, indonesia_to_bebasan, ngoko_to_indonesia, indonesia_to_ngoko = process_file(file_path)
            
            # Merge the results into the final dictionaries
            for key, value in ngoko_to_bebasan.items():
                ngoko_to_bebasan_all[key].extend(value)
            for key, value in bebasan_to_ngoko.items():
                bebasan_to_ngoko_all[key].extend(value)
            for key, value in bebasan_to_indonesia.items():
                bebasan_to_indonesia_all[key].extend(value)
            for key, value in indonesia_to_bebasan.items():
                indonesia_to_bebasan_all[key].extend(value)
            for key, value in ngoko_to_indonesia.items():
                ngoko_to_indonesia_all[key].extend(value)
            for key, value in indonesia_to_ngoko.items():
                indonesia_to_ngoko_all[key].extend(value)

    return ngoko_to_bebasan_all, bebasan_to_ngoko_all, bebasan_to_indonesia_all, indonesia_to_bebasan_all, ngoko_to_indonesia_all, indonesia_to_ngoko_all

def process_file_five_columns(file_path, translation_dict):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Skip the first line
        for _ in range(1):
            next(file)
        
        for line in file:
            # Clean the line (remove extra spaces and split by '\t')
            parts = [part.strip() for part in line.split('\t')]
            
            if len(parts) > 5:
                parts = parts[:5]
            elif len(parts) < 5:
                print("PARTS:", parts, "; LINE:", line)
                continue
            
            # Clean the words by removing parentheses content
            ngoko_word = clean_word(parts[0])
            ngoko_word_ = clean_word(parts[2])
            bebasan_word = clean_word(parts[1])
            bebasan_word_ = clean_word(parts[3])
            indonesia_word = clean_word(parts[4])
            
            # Handle cases where there are multiple translations separated by commas
            ngoko_words = list(set([word.strip().lower() for word in split_comma_slash_string(ngoko_word) + split_comma_slash_string(ngoko_word_)]))
            bebasan_words = list(set([word.strip().lower() for word in split_comma_slash_string(bebasan_word) + split_comma_slash_string(bebasan_word_)]))
            indonesia_words = list(set([word.strip().lower() for word in split_comma_slash_string(indonesia_word)]))

            ngoko_words = clean_list(ngoko_words)
            bebasan_words = clean_list(bebasan_words)
            indonesia_words = clean_list(indonesia_words)

            # Add translations to dictionaries
            for ngoko in ngoko_words:
                for bebasan in bebasan_words:
                    if ngoko not in translation_dict["cbn_ngoko-cbn_bebasan"]:
                        translation_dict["cbn_ngoko-cbn_bebasan"][ngoko] = [bebasan]
                    else:
                        translation_dict["cbn_ngoko-cbn_bebasan"][ngoko].append(bebasan)
                    if bebasan not in translation_dict["cbn_bebasan-cbn_ngoko"]:
                        translation_dict["cbn_bebasan-cbn_ngoko"][bebasan] = [ngoko]
                    else:
                        translation_dict["cbn_bebasan-cbn_ngoko"][bebasan].append(ngoko)
                for indonesia in indonesia_words:
                    if ngoko not in translation_dict["cbn_ngoko-indonesia"]:
                        translation_dict["cbn_ngoko-indonesia"][ngoko] = [indonesia]
                    else:
                        translation_dict["cbn_ngoko-indonesia"][ngoko].append(indonesia)
                    if indonesia not in translation_dict["indonesia-cbn_ngoko"]:
                        translation_dict["indonesia-cbn_ngoko"][indonesia] = [ngoko]
                    else:
                        translation_dict["indonesia-cbn_ngoko"][indonesia].append(ngoko)

            for bebasan in bebasan_words:
                for indonesia in indonesia_words:
                    if bebasan not in translation_dict["cbn_bebasan-indonesia"]:
                        translation_dict["cbn_bebasan-indonesia"][bebasan] = [indonesia]
                    else:
                        translation_dict["cbn_bebasan-indonesia"][bebasan].append(indonesia)
                    if indonesia not in translation_dict["indonesia-cbn_bebasan"]:
                        translation_dict["indonesia-cbn_bebasan"][indonesia] = [bebasan]
                    else:
                        translation_dict["indonesia-cbn_bebasan"][indonesia].append(bebasan)
        
        return translation_dict

# Main entry point
if __name__ == "__main__":
    folder_path = './dict/kamus-indramayu/'  # Replace with your folder path containing txt files
    ngoko_to_bebasan, bebasan_to_ngoko, bebasan_to_indonesia, indonesia_to_bebasan, ngoko_to_indonesia, indonesia_to_ngoko = process_files_in_folder(folder_path)

    # Combine ngoko and bebasan dictionaries into cirebonese dictionaries
    cirebonese_to_indonesia = defaultdict(list)
    indonesia_to_cirebonese = defaultdict(list)

    # Merge ngoko_to_indonesia and bebasan_to_indonesia into cirebonese_to_indonesia
    for key, value in ngoko_to_indonesia.items():
        cirebonese_to_indonesia[key].extend(value)
    for key, value in bebasan_to_indonesia.items():
        cirebonese_to_indonesia[key].extend(value)

    # Merge indonesia_to_ngoko and indonesia_to_bebasan into indonesia_to_cirebonese
    for key, value in indonesia_to_ngoko.items():
        indonesia_to_cirebonese[key].extend(value)
    for key, value in indonesia_to_bebasan.items():
        indonesia_to_cirebonese[key].extend(value)

    # Clean the combined dictionaries
    cirebonese_to_indonesia = {k: list(set(v)) for k, v in cirebonese_to_indonesia.items() if k}
    indonesia_to_cirebonese = {k: list(set(v)) for k, v in indonesia_to_cirebonese.items() if k}

    # Optionally, you can save the dictionaries to JSON files
    cirebonese_to_indonesia_file = './dict/cbn_idn2.json'
    indonesia_to_cirebonese_file = './dict/idn_cbn2.json'

    # Save the cirebonese_to_indonesia dictionary to a JSON file
    with open(cirebonese_to_indonesia_file, 'w', encoding='utf-8') as json_file:
        json.dump(cirebonese_to_indonesia, json_file, ensure_ascii=False, indent=4)

    # Save the indonesia_to_cirebonese dictionary to a JSON file
    with open(indonesia_to_cirebonese_file, 'w', encoding='utf-8') as json_file:
        json.dump(indonesia_to_cirebonese, json_file, ensure_ascii=False, indent=4)

    print(f"Saved the dictionaries to {cirebonese_to_indonesia_file} and {indonesia_to_cirebonese_file}")

    # for key in cirebonese_to_indonesia.keys():
    #     print("Cirebonese to Indonesia KEY:", key, "; NUM KEYS:", len(cirebonese_to_indonesia[key]))
    # for key in indonesia_to_cirebonese.keys():
    #     print("Indonesia to Cirebonese KEY:", key, "; NUM KEYS:", len(indonesia_to_cirebonese[key]))
