import json
import re

def fix_word_splitting(text):
    if not text:
        return text
    
    # Step 1: Remove extra spaces within words
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Step 2: Split camelCase-like concatenations (e.g., "Providespecialized" -> "Provide specialized")
    def split_camel_case(match):
        word = match.group(0)
        # Insert space before any uppercase letter that follows a lowercase letter
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
    
    # Apply camelCase splitting to each word
    words = text.split()
    words = [re.sub(r'[a-zA-Z]+', split_camel_case, word) for word in words]
    text = ' '.join(words)
    
    # Step 3: Clean up any remaining double spaces
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def clean_contractor_name(name):
    if not name:
        return name
    # Remove extra spaces and ensure proper capitalization
    words = re.split(r'\s+', name.strip())
    # Filter out empty strings and capitalize each word
    words = [word.capitalize() for word in words if word]
    return ' '.join(words)

def clean_contract_data(input_file, output_file):
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            contracts = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_file}.")
        return
    
    cleaned_contracts = []
    
    for contract in contracts:
        # Define fields to keep with default values for missing fields
        relevant_fields = {
            'language': contract.get('translated_language', 'english'),
            'translation_timestamp': contract.get('translation_timestamp', ''),
            'translation_method': contract.get('translation_method', ''),
            'contract_object': fix_word_splitting(contract.get('contract_object', '')),
            'process_object': fix_word_splitting(contract.get('process_object', '')),
            'contract_modality': contract.get('contract_modality', ''),
            'contract_type': contract.get('contract_type', ''),
            'process_status': fix_word_splitting(contract.get('process_status', '')),
            'entity_department': contract.get('entity_department', ''),
            'entity_municipality': fix_word_splitting(contract.get('entity_municipality', '')),
            'contractor_name': clean_contractor_name(contract.get('contractor_name', '')),
            'contract_value': contract.get('contract_value', ''),
            'execution_end_date': contract.get('execution_end_date', ''),
            'contract_number': contract.get('contract_number', ''),
            'process_number': contract.get('process_number', ''),
            'entity_nit': contract.get('entity_nit', ''),
            'entity_code': contract.get('entity_code', ''),
            'provider_document': contract.get('provider_document', ''),
            'contract_url': contract.get('contract_url', ''),
            'search_keyword': contract.get('search_keyword', ''),
            'collection_time': contract.get('collection_time', ''),
            'improvement_timestamp': contract.get('improvement_timestamp', '')
        }
        
        cleaned_contracts.append(relevant_fields)
    
    try:
        # Save cleaned data to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_contracts, f, indent=2, ensure_ascii=False)
        print(f"Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error saving output file: {str(e)}")

if __name__ == "__main__":
    input_file = "data/healthcare_enhanced_20250718_181121_streamlined_20250718_182651.json"  # Replace with your input file path
    output_file = "data/cleaned_contracts_healthcare.json"  # Output file path
    clean_contract_data(input_file, output_file)