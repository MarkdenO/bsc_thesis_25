import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


def extract_labels(df, year, day):
    filtered_row = df[(df['Year'] == year) & (df['Day'] == day)]

    if filtered_row.empty:
        return []

    specific_row_series = filtered_row.iloc[0]
    x_marked_columns = specific_row_series[specific_row_series == 'X']
    
    return x_marked_columns.index.tolist()


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
extra_repos_data = []


for year in YEARS:
    GH_FILENAME = f"lb_{year}.json"
    GH_FILEPATH = os.path.join(BASE_PATH, GH_FILENAME)
    REDDIT_FILENAME = f"reddit_{year}.json"
    REDDIT_FILEPATH = os.path.join(BASE_PATH, REDDIT_FILENAME)
    EXTRA_REPOS_FILENAME = f"extra_repos_{year}.json"
    EXTRA_REPOS_FILEPATH = os.path.join(BASE_PATH, EXTRA_REPOS_FILENAME)

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
                            'DataSource': 'leaderboard',
                            'Data': code_data,
                            'Labels': extract_labels(labels_df, year, day_int)
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
                            'DataSource': 'reddit',
                            'Data': code_data,
                            'Labels': extract_labels(labels_df, year, day_int)
                        }
                    reddit_code_data.append(record)
        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON from {REDDIT_FILEPATH}: {e}")
        except Exception as e:
            print(f"  Error processing file {REDDIT_FILEPATH}: {e}")
    else:
        print(f"File not found: {REDDIT_FILEPATH}")
    
    if os.path.exists(EXTRA_REPOS_FILEPATH):
        print(f"Processing: {EXTRA_REPOS_FILEPATH}...")
        try:
            with open(EXTRA_REPOS_FILEPATH, 'r', encoding='utf-8') as f:
                data_year = json.load(f)

            for day_str, sources_dict in data_year.items():
                try:
                    day_int = int(day_str)
                except ValueError:
                    print(f"  Warning: Invalid day format '{day_str}' in {EXTRA_REPOS_FILEPATH}. Skipping.")
                    continue

                if not isinstance(sources_dict, dict):
                    print(f"  Warning: Expected dict for day '{day_str}', found {type(sources_dict)} in {EXTRA_REPOS_FILEPATH}. Skipping day.")
                    continue

                for source, code_data in sources_dict.items():
                    record = {
                            'Year': year,
                            'Day': day_int,
                            'DataSource': 'github',
                            'Data': code_data,
                            'Labels': extract_labels(labels_df, year, day_int)
                    }
                    extra_repos_data.append(record)
        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON from {GH_FILEPATH}: {e}")
        except Exception as e:
            print(f"  Error processing file {GH_FILEPATH}: {e}")
    else:
        print(f"File not found: {GH_FILEPATH}")

# Convert to DataFrame
if not github_code_data and not reddit_code_data:
    print("No code data was extracted. Cannot create DataFrame.")
    exit()

github_code_df = pd.DataFrame(github_code_data)
reddit_code_df = pd.DataFrame(reddit_code_data)
extra_repos_df = pd.DataFrame(extra_repos_data)
github_code_df['Year'] = github_code_df['Year'].astype(int)
github_code_df['Day'] = github_code_df['Day'].astype(int)
reddit_code_df['Year'] = reddit_code_df['Year'].astype(int)
reddit_code_df['Day'] = reddit_code_df['Day'].astype(int)
extra_repos_df['Year'] = extra_repos_df['Year'].astype(int)
extra_repos_df['Day'] = extra_repos_df['Day'].astype(int)

gh_labelled_df = pd.merge(labels_df, github_code_df, on=['Year', 'Day'], how='left')
reddit_labelled_df = pd.merge(labels_df, reddit_code_df, on=['Year', 'Day'], how='left')
extra_repos_labelled_df = pd.merge(labels_df, extra_repos_df, on=['Year', 'Day'], how='left')
final_df = pd.concat([gh_labelled_df, reddit_labelled_df, extra_repos_labelled_df], ignore_index=True)
print(f"Created final DataFrame. Shape: {final_df.shape}")

print("\nFinal DataFrame Info:")
final_df.info()
print("\nFinal DataFrame Head:")
print(final_df.head())

output_csv_path = 'labelled_data.csv'
final_df.to_csv(output_csv_path, index=False)
print(f"Final DataFrame saved to {output_csv_path}")

output_pkl_path = 'labelled_data.pkl'
final_df.to_pickle(output_pkl_path)
print(f"Final DataFrame saved to {output_pkl_path}")


def split_labelled_data(df, output_dir='datasets', seed=42):
    os.makedirs(output_dir, exist_ok=True)

    df = df[['Year', 'Day', 'DataSource', 'Data', 'Labels']]

    # Remove rows with missing or empty Data
    df = df[df['Data'].notnull() & (df['Data'].str.strip() != '')]

    # Split the data: 80% train, 10% val, 10% test
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=seed)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=seed)

    print(f"\nData split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Save as JSON
    train_df.to_json(os.path.join(output_dir, 'train.json'), orient='records', indent=2)
    val_df.to_json(os.path.join(output_dir, 'val.json'), orient='records', indent=2)
    test_df.to_json(os.path.join(output_dir, 'test.json'), orient='records', indent=2)
    print(f"JSON splits saved in '{output_dir}'")

    # Create seperate test.json files for each data source
    for data_source in df['DataSource'].unique():
        source_test_df = test_df[test_df['DataSource'] == data_source]
        source_test_df.to_json(os.path.join(output_dir, f'test_{data_source}.json'), orient='records', indent=2)
        print(f"Test split for '{data_source}' saved as 'test_{data_source}.json'")

    # Split test data into single-label and multi-label
    single_test_df = test_df[test_df['Labels'].apply(lambda x: len(x) == 1 if isinstance(x, list) else False)]
    multi_test_df = test_df[test_df['Labels'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False)]
    single_test_df.to_json(os.path.join(output_dir, 'test_single.json'), orient='records', indent=2)
    multi_test_df.to_json(os.path.join(output_dir, 'test_multi.json'), orient='records', indent=2)
    print("Test data split into single-label and multi-label saved as 'test_single.json' and 'test_multi.json'")

split_labelled_data(final_df)
