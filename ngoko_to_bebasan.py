import json
import random

def load_dictionary(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def translate_to_bebasan(ngoko_text, translations):
    words = ngoko_text.strip().split()
    translated_words = []
    changed_words = {}
    
    for word in words:
        # Check if word exists in ngoko-bebasan dictionary
        if word in translations['cbn_ngoko-cbn_bebasan']:
            # Get random translation from available options
            translated_word = random.choice(translations['cbn_ngoko-cbn_bebasan'][word])
            translated_words.append(translated_word)
            changed_words[word] = translated_word
            continue
            
        # If word not found, keep original
        translated_words.append(word)
    
    return {
        'bebasan_text': ' '.join(translated_words),
        'ngoko_to_bebasan_changes': changed_words
    }

def add_bebasan_translations(input_file, output_file, translations):
    # Read existing translations
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add bebasan translations
    for item in data:
        bebasan_result = translate_to_bebasan(item['ngoko_text'], translations)
        item.update(bebasan_result)
    
    # Write updated results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Usage example:
translations = load_dictionary('translations.json')
add_bebasan_translations('output_translations.json', 'output_translations_with_bebasan.json', translations)