import pandas as pd
import json
import os
import glob

# Configuration
BASE_PATH = 'results'
CSV_PATH = 'categorization.csv'
DATA_TYPES = ['ast', 'cfg', 'embed', 'ngrams', 'tfidf']
DATA_SOURCES = ['github', 'reddit']
YEARS = range(2015, 2024)

# Load labels
try:
    labels_df = pd.read_csv(CSV_PATH)
    labels_df['Year'] = labels_df['Year'].astype(int)
    labels_df['Day'] = labels_df['Day'].astype(int)
    print(f"Loaded labels from {CSV_PATH}. Shape: {labels_df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find the CSV file at {CSV_PATH}")
    exit()
except Exception as e:
    print(f"Error reading CSV file {CSV_PATH}: {e}")
    exit()


# Iterate and Extract JSON Data
all_code_data = []

for data_type in DATA_TYPES:
    for data_source in DATA_SOURCES:
        for year in YEARS:
            json_filename = f"{year}.json"
            json_filepath = os.path.join(BASE_PATH, data_type, data_source, json_filename)

            if os.path.exists(json_filepath):
                print(f"Processing: {json_filepath}...")
                try:
                    with open(json_filepath, 'r', encoding='utf-8') as f:
                        data_year = json.load(f)

                    for day_str, sources_dict in data_year.items():
                        try:
                            day_int = int(day_str)
                        except ValueError:
                            print(f"  Warning: Invalid day format '{day_str}' in {json_filepath}. Skipping.")
                            continue

                        if not isinstance(sources_dict, dict):
                            print(f"  Warning: Expected dict for day '{day_str}', found {type(sources_dict)} in {json_filepath}. Skipping day.")
                            continue

                        for source, code_data in sources_dict.items():
                            record = {
                                'Year': year,
                                'Day': day_int,
                                'Source': source,
                                'DataSource': data_source,
                                'DataType': data_type,
                                'Data': code_data
                            }
                            all_code_data.append(record)

                except json.JSONDecodeError as e:
                    print(f"  Error decoding JSON from {json_filepath}: {e}. Skipping file.")
                except Exception as e:
                    print(f"  Unexpected error processing {json_filepath}: {e}. Skipping file.")
            else:
                pass

print(f"\nExtracted {len(all_code_data)} code samples.")

# Create Code DataFrame ---
if not all_code_data:
    print("No code data was extracted. Cannot create DataFrame.")
    exit()

code_df = pd.DataFrame(all_code_data)
code_df['Year'] = code_df['Year'].astype(int)
code_df['Day'] = code_df['Day'].astype(int)
print(f"Created code DataFrame. Shape: {code_df.shape}")

# Merge dfs ---
final_df = pd.merge(labels_df, code_df, on=['Year', 'Day'], how='left')

print(f"\nMerged DataFrame created. Shape: {final_df.shape}")

# df info
print("\nFinal DataFrame Info:")
final_df.info()

print("\nFinal DataFrame Head:")
print(final_df.head())

# Save df to csv and pkl
final_df.to_csv('merged_aoc_data.csv', index=False)
final_df.to_pickle('merged_aoc_data.pkl')