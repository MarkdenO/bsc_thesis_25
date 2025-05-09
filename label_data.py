import pandas as pd
import json
import os

BASE_PATH = 'preprocessed_code'
CSV_PATH = 'categorization.csv'

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

github_code_data = []
reddit_code_data = []


for year in YEARS:
    GH_FILENAME = f"lb_{year}.json"
    GH_FILEPATH = os.path.join(BASE_PATH, GH_FILENAME)
    REDDIT_FILENAME = f"reddit_{year}.json"
    REDDIT_FILEPATH = os.path.join(BASE_PATH, REDDIT_FILENAME)

    if os.path.exists(GH_FILEPATH):
        print(f"Processing: {GH_FILEPATH}...")
        try:
            with open(GH_FILEPATH, 'r', encoding='utf-8') as f:
                data_year = json.load(f)

            for day_str, sources_dict in data_year.items():
                try:
                    day_int = int(day_str)
                except ValueError:
                    print(f"  Warning: Invalid day format '{day_str}' in {GH_FILEPATH}. Skipping.")
                    continue

                if not isinstance(sources_dict, dict):
                    print(f"  Warning: Expected dict for day '{day_str}', found {type(sources_dict)} in {GH_FILEPATH}. Skipping day.")
                    continue

                for source, code_data in sources_dict.items():
                    record = {
                        'Year': year,
                        'Day': day_int,
                        'Source': source,
                        'DataSource': 'github',
                        'Data': code_data
                    }
                    github_code_data.append(record)
        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON from {GH_FILEPATH}: {e}")
        except Exception as e:
            print(f"  Error processing file {GH_FILEPATH}: {e}")
    else:
        print(f"File not found: {GH_FILEPATH}")
    if os.path.exists(REDDIT_FILEPATH):
        print(f"Processing: {REDDIT_FILEPATH}...")
        try:
            with open(REDDIT_FILEPATH, 'r', encoding='utf-8') as f:
                data_year = json.load(f)

            for day_str, sources_dict in data_year.items():
                try:
                    day_int = int(day_str)
                except ValueError:
                    print(f"  Warning: Invalid day format '{day_str}' in {REDDIT_FILEPATH}. Skipping.")
                    continue

                if not isinstance(sources_dict, dict):
                    print(f"  Warning: Expected dict for day '{day_str}', found {type(sources_dict)} in {REDDIT_FILEPATH}. Skipping day.")
                    continue

                for source, code_data in sources_dict.items():
                    record = {
                        'Year': year,
                        'Day': day_int,
                        'Source': source,
                        'DataSource': 'reddit',
                        'Data': code_data
                    }
                    reddit_code_data.append(record)
        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON from {REDDIT_FILEPATH}: {e}")
        except Exception as e:
            print(f"  Error processing file {REDDIT_FILEPATH}: {e}")
    else:
        print(f"File not found: {REDDIT_FILEPATH}")

# Convert to DataFrame
if not github_code_data and not reddit_code_data:
    print("No code data was extracted. Cannot create DataFrame.")
    exit()

github_code_df = pd.DataFrame(github_code_data)
reddit_code_df = pd.DataFrame(reddit_code_data)
github_code_df['Year'] = github_code_df['Year'].astype(int)
github_code_df['Day'] = github_code_df['Day'].astype(int)
reddit_code_df['Year'] = reddit_code_df['Year'].astype(int)
reddit_code_df['Day'] = reddit_code_df['Day'].astype(int)

final_df = pd.merge(labels_df, github_code_df, on=['Year', 'Day'], how='left')
final_df = pd.merge(final_df, reddit_code_df, on=['Year', 'Day'], how='left', suffixes=('_github', '_reddit'))
print(f"Created final DataFrame. Shape: {final_df.shape}")

print("\nFinal DataFrame Info:")
final_df.info()
print("\nFinal DataFrame Head:")
print(final_df.head())

output_csv_path = 'final_output.csv'
final_df.to_csv(output_csv_path, index=False)
print(f"Final DataFrame saved to {output_csv_path}")

output_pkl_path = 'final_output.pkl'
final_df.to_pickle(output_pkl_path)
print(f"Final DataFrame saved to {output_pkl_path}")
