import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Base directory containing Cancer and Non-Cancer folders
base_dir = os.path.join(os.getcwd(), "Dataset")  
categories = ['Cancer', 'Non-Cancer']

print(f"Looking for {categories} folders in {base_dir}")

data = []

# Parse function to extract id, title and abstract from text files
def parse_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    lines = content.split('\n')
    id_line = next((line for line in lines if line.startswith('<ID:')), None)
    title_line = next((line for line in lines if line.startswith('Title:')), None)
    abstract_line = next((line for line in lines if line.startswith('Abstract:')), None)

    id_val = id_line.replace('<ID:', '').replace('>', '').strip() if id_line else ''
    title_val = title_line.replace('Title:', '').strip() if title_line else ''
    abstract_val = abstract_line.replace('Abstract:', '').strip() if abstract_line else ''
    return id_val, title_val, abstract_val

# Read all files from Cancer and Non-Cancer folders
for category in categories:
    folder = os.path.join(base_dir, category)
    
    if not os.path.exists(folder):
        print(f"Warning: {folder} not found!")
        continue
        
    print(f"Processing {category} files from {folder}")
    num_files = 0
    
    for fname in os.listdir(folder):
        if fname.endswith('.txt') and not fname.startswith('._'):
            fpath = os.path.join(folder, fname)
            id_val, title, abstract = parse_text_file(fpath)
            if abstract:
                # For Cancer folder, label=1; for Non-Cancer folder, label=0
                label = 1 if category == 'Cancer' else 0
                data.append({
                    "id": id_val,
                    "title": title,
                    "abstract": abstract,
                    "filename": fname,
                    "label": label
                })
                num_files += 1
    
    print(f"  Processed {num_files} files from {category} folder")

# Create DataFrame and clean data
print("Creating dataset...")
df = pd.DataFrame(data)
df = df.dropna(subset=['abstract'])

# Ensure we have the right columns in the right order
expected_columns = ['id', 'title', 'abstract', 'filename', 'label']
for col in expected_columns:
    if col not in df.columns:
        print(f"Warning: Adding missing column {col}")
        df[col] = ''

# Reorder columns to match existing CSV format
df = df[expected_columns]

print(f"Total records: {len(df)}")

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train-val-test split (80-10-10)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# Save CSVs to Dataset/data directory
output_dir = os.path.join(base_dir, "data")
os.makedirs(output_dir, exist_ok=True)

# Save with the correct column names and order
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print("✅ Data preparation complete.")
print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
print(f"Files saved to {output_dir} with columns: {', '.join(expected_columns)}")
