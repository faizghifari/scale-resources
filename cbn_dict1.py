import glob
import os
import json

def build_cirebonese_dict_from_file(file_path):
    """
    Reads a file containing Cirebonese-to-Indonesian translations and
    builds two dictionaries: one where the key is the Cirebonese word and the
    value is a list of Indonesian translation words, and another where the key
    is the Indonesian word and the value is a list of Cirebonese translation words.

    Args:
        file_path (str): Path to the input text file.

    Returns:
        tuple: Two dictionaries, one for Cirebonese-to-Indonesian and one for
               Indonesian-to-Cirebonese translations.
    """
    # Initialize the dictionaries
    cirebonese_to_indonesian_dict = {}
    indonesian_to_cirebonese_dict = {}

    # Open and read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # Read all lines into a list

    # Iterate through each line to process key-value pairs
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace or newlines
        if line and '|' in line:  # Process only lines containing the pipe symbol '|'
            # Split the line into Cirebonese and Indonesian parts
            cirebonese_part, indonesian_part = line.split('|', 1)

            # Clean up and split Cirebonese and Indonesian words
            cirebonese_words = [word.strip() for part in cirebonese_part.split(';') for word in part.split(',')]
            indonesian_words = [word.strip() for part in indonesian_part.split(';') for word in part.split(',')]

            # Add each Cirebonese word to the Cirebonese-to-Indonesian dictionary with its translations
            for word in cirebonese_words:
                if word not in cirebonese_to_indonesian_dict.keys():
                    cirebonese_to_indonesian_dict[word] = indonesian_words
                else:
                    cirebonese_to_indonesian_dict[word].extend(indonesian_words)

            # Add each Indonesian word to the Indonesian-to-Cirebonese dictionary with its translations
            for word in indonesian_words:
                if word not in indonesian_to_cirebonese_dict.keys():
                    indonesian_to_cirebonese_dict[word] = cirebonese_words
                else:
                    indonesian_to_cirebonese_dict[word].extend(cirebonese_words)

    return cirebonese_to_indonesian_dict, indonesian_to_cirebonese_dict

# Directory containing the text files
directory_path = "./dict/Kamus Bahasa Cirebon/actual_pages/"

# Get all .txt files in the directory
txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

# Initialize the combined dictionaries
combined_cirebonese_to_indonesian_dict = {}
combined_indonesian_to_cirebonese_dict = {}

# Process each file and build dictionaries
for txt_file in txt_files:
    cirebonese_to_indonesian_dict, indonesian_to_cirebonese_dict = build_cirebonese_dict_from_file(txt_file)
    
    # Merge the current dictionaries with the combined dictionaries
    for key, value in cirebonese_to_indonesian_dict.items():
        if key and key in combined_cirebonese_to_indonesian_dict:
            combined_cirebonese_to_indonesian_dict[key].extend(value)
        elif key:
            combined_cirebonese_to_indonesian_dict[key] = value

    for key, value in indonesian_to_cirebonese_dict.items():
        if key and key in combined_indonesian_to_cirebonese_dict:
            combined_indonesian_to_cirebonese_dict[key].extend(value)
        elif key:
            combined_indonesian_to_cirebonese_dict[key] = value

# Ensure each key has a unique list of translations and remove empty strings
for key in list(combined_cirebonese_to_indonesian_dict.keys()):
    combined_cirebonese_to_indonesian_dict[key] = list(set(filter(None, combined_cirebonese_to_indonesian_dict[key])))
    if not combined_cirebonese_to_indonesian_dict[key]:
        del combined_cirebonese_to_indonesian_dict[key]

for key in list(combined_indonesian_to_cirebonese_dict.keys()):
    combined_indonesian_to_cirebonese_dict[key] = list(set(filter(None, combined_indonesian_to_cirebonese_dict[key])))
    if not combined_indonesian_to_cirebonese_dict[key]:
        del combined_indonesian_to_cirebonese_dict[key]

# Write the combined dictionaries to JSON files
# combined_cirebonese_to_indonesian_json_file_path = os.path.join(directory_path, "combined_cirebonese_to_indonesian_dict.json")
combined_cirebonese_to_indonesian_json_file_path = "./dict/cbn_idn1.json"
with open(combined_cirebonese_to_indonesian_json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(combined_cirebonese_to_indonesian_dict, json_file, ensure_ascii=False, indent=4)

# combined_indonesian_to_cirebonese_json_file_path = os.path.join(directory_path, "combined_indonesian_to_cirebonese_dict.json")
combined_indonesian_to_cirebonese_json_file_path = "./dict/idn_cbn1.json"
with open(combined_indonesian_to_cirebonese_json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(combined_indonesian_to_cirebonese_dict, json_file, ensure_ascii=False, indent=4)