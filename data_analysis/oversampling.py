import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import ast
import numpy as np

target_percentages = {'DAT': 0.25,
                      'LOC': 0.20,
                      'PER': 0.18,
                      'ORG': 0.20,
                      'EVE': 0.17}
merged_label_map = {
    'B-PER': 'PER', 'I-PER': 'PER',
    'B-ORG': 'ORG', 'I-ORG': 'ORG',
    'B-LOC': 'LOC', 'I-LOC': 'LOC',
    'B-DAT': 'DAT', 'I-DAT': 'DAT',
    'B-EVE': 'EVE', 'I-EVE': 'EVE'
}

LABEL2IDX = {
    'O': 0,
    'B-DAT': 1,
    'B-PER': 2,
    'B-ORG': 3,
    'B-LOC': 4,
    'B-EVE': 5,
    'I-DAT': 6,
    'I-PER': 7,
    'I-ORG': 8,
    'I-LOC': 9,
    'I-EVE': 10
}

# Reverse the dictionary to get IDX2LABEL for easy lookup
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


def clean_labels(label_str):
    try:
        # Convert string representation of list to actual list
        label_list = ast.literal_eval(label_str)
        # Ensure the labels are integers
        label_list = [int(label) for label in label_list]
        return label_list
    except (ValueError, SyntaxError):
        # Return an empty list or handle the error as needed
        return []


def count_each_lab(df: pd.DataFrame) -> Counter:
    all_labels = []
    for labels in df['labels']:
        all_labels.extend(labels)

    # Convert numerical labels to human-readable form and merge using the above mapping
    readable_labels_merged = [merged_label_map.get(IDX2LABEL[label], IDX2LABEL[label]) for label in all_labels if
                              label != LABEL2IDX['O']]

    # Count the occurrences of each merged label
    return Counter(readable_labels_merged)


# Function to calculate how close the current sample is to the target
def calculate_distance(counts, target, total_count, penalty_factor=5):
    label_percentages = {label: (count / total_count) * 100 for label, count in counts.items()}
    distance = 0
    for label in target:
        # Calculate the distance for the current label
        label_distance = abs(label_percentages.get(label, 0) - target[label] * 100)
        # Apply a penalty factor if the label is 'EVE'
        if label == 'DAT':
            label_distance *= (penalty_factor * 2)
        if label == 'EVE':
            label_distance *= (penalty_factor * 1)
        distance += label_distance
    return distance, label_percentages


def incremental_oversample(df, df_size, target, step_size=100, max_distance=10, restart_threshold=100,
                           max_restarts=1000):
    current_df = pd.DataFrame(columns=df.columns)
    remaining_df = df.copy()

    best_distance = float('inf')
    best_df = current_df
    restarts = 0
    last_EVE = 0

    while restarts < max_restarts:
        iterations_without_improvement = 0
        last_DAT = 0
        while len(current_df) < df_size:
            num_to_sample = min(step_size, df_size - len(current_df))
            sampled_df = remaining_df.sample(n=num_to_sample, replace=False)
            temp_df = pd.concat([current_df, sampled_df], ignore_index=True)

            counts = count_each_lab(temp_df)
            total_count = sum(counts.values())
            distance, lp = calculate_distance(counts, target, total_count)
            if distance < best_distance and np.round(lp['EVE'], 2) > np.round(last_EVE, 2):
                best_distance = distance
                best_df = temp_df.copy()
                current_df = temp_df
                remaining_df = remaining_df.drop(sampled_df.index)
                last_EVE = lp['EVE']
                last_DAT = lp['DAT']
                print(f"Updated current size: {len(current_df)}, Current distance: {distance}")
                print(lp)
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            if best_distance <= max_distance and len(current_df) >= df_size:
                print("Desired distribution achieved.")
                print(lp)
                return best_df
            # Check if a restart is needed
            if iterations_without_improvement >= restart_threshold:
                print(f"Restarting the algorithm. Restart count: {restarts + 1}")
                current_df = best_df.copy()  # Start over from the best found so far
                remaining_df = df.drop(current_df.index)  # Reset remaining samples
                restarts += 1
                break
    while len(current_df) < df_size:
        # Determine the number of rows to add in this step
        num_to_sample = min(step_size * 2, df_size - len(current_df))
        # Sample rows from the remaining DataFrame
        sampled_df = remaining_df.sample(n=num_to_sample, replace=False)
        # Temporarily concatenate the sampled rows to the current dataset
        temp_df = pd.concat([current_df, sampled_df], ignore_index=True)
        # Count the current label distribution in the temporary DataFrame
        counts = count_each_lab(temp_df)
        total_count = sum(counts.values())
        # Calculate the current distance to the target distribution
        distance, lp = calculate_distance(counts, target, total_count)
        # Only update the current dataset if the new distance is smaller
        if distance < best_distance:
            best_distance = distance
            best_df = temp_df.copy()
            current_df = temp_df  # Keep the improved dataset
            remaining_df = remaining_df.drop(sampled_df.index)  # Remove added samples from remaining data
            print(f"Updated current size: {len(current_df)}, Current distance: {distance}")
            print(lp)
        # else:
        # print(f"Skipped adding sample, distance not improved: {distance}")
        # Check if the distance is within the acceptable range
        if len(current_df) >= df_size:
            print("Desired distribution achieved.")
            break
    return best_df


test_size = eval_size = 3000
train_size = 24000

merged_df = pd.read_csv('merged_data.csv')
# Apply the cleaning function to the 'labels' column
merged_df['labels'] = merged_df['labels'].apply(clean_labels)

# Re-extract unique labels after cleaning
unique_labels_cleaned = set()
for labels in merged_df['labels']:
    unique_labels_cleaned.update(labels)

# Print unique labels after cleaning
print("Unique labels in the dataframe after cleaning:")
print(unique_labels_cleaned)

train_eval_df, test_df = train_test_split(merged_df, test_size=test_size, random_state=42)
train_df, eval_df = train_test_split(train_eval_df, test_size=eval_size, random_state=42)

train_df_oversampled = incremental_oversample(merged_df, df_size=train_size, target=target_percentages)
train_df_oversampled.to_csv('stratified/train.csv', index=False)
test_df.to_csv('stratified/test.csv', index=False)
eval_df.to_csv('stratified/eval.csv', index=False)
