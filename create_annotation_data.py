import json
import random

def process_json_file(file_path, output_path):
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Get all texts from all topics
    all_texts = []
    for texts in data.values():
        all_texts.extend(texts)
    
    # Split texts into sentences and filter by word count
    sentences = []
    for text in all_texts:
        # Split by period and clean each sentence
        text_sentences = [s.strip() for s in text.split('.')]
        # Filter out empty sentences and sentences with less than 10 words
        valid_sentences = [s for s in text_sentences 
                         if s and len(s.split()) >= 10]
        sentences.extend(valid_sentences)
    
    # Choose 200 random sentences
    # If there are less than 200 valid sentences, take all available
    # num_sentences = min(200, len(sentences))
    # random_sentences = random.sample(sentences, num_sentences)
    
    # Save to text file
    with open(output_path, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(f"{sentence}\n")
    
    return len(sentences)

# Example usage
file_path = './direct/topics_2.json'
output_path = 'raw.txt'
try:
    num_sentences = process_json_file(file_path, output_path)
    print(f"Successfully saved {num_sentences} sentences to {output_path}")
except FileNotFoundError:
    print("File not found!")
except json.JSONDecodeError:
    print("Invalid JSON file!")