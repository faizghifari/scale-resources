import json
import random

def load_dictionary(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def translate_sentence(sentence, translations):
    # Split sentence into words
    words = sentence.strip().split()
    translated_words = []
    changed_words = {}
    
    for raw_word in words:
        word = raw_word.lower()
        # Check if word exists in cbn_ngoko (by checking indonesia-cbn_ngoko dictionary)
        ngoko_exists = False
        for ngoko_word in translations['indonesia-cbn_ngoko'].values():
            if word in ngoko_word:
                ngoko_exists = True
                translated_words.append(raw_word)
                break
                
        if ngoko_exists:
            continue
            
        # Check if word exists in indonesia dictionary
        if word in translations['indonesia-cbn_ngoko']:
            # Get random translation from available options
            translated_word = random.choice(translations['indonesia-cbn_ngoko'][word])
            translated_words.append(translated_word)
            changed_words[word] = translated_word
            continue
            
        # Check if word exists in bebasan dictionary
        if word in translations['cbn_bebasan-cbn_ngoko']:
            # Get random translation from available options
            translated_word = random.choice(translations['cbn_bebasan-cbn_ngoko'][word])
            translated_words.append(translated_word)
            changed_words[word] = translated_word
            continue
            
        # If word not found in any dictionary, keep original
        translated_words.append(raw_word)
    
    return {
        'raw_text': sentence.strip(),
        'ngoko_text': ' '.join(translated_words),
        'changed_words': changed_words
    }

def translate_file(input_file, output_file, translations):
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            translation_result = translate_sentence(line, translations)
            if bool(translation_result['changed_words']):
                results.append(translation_result)
    
    # Write results to JSON file
    print(len(results))
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)

# Usage example:
translations = load_dictionary('translations.json')
translate_file('raw.txt', 'output_translations.json', translations)