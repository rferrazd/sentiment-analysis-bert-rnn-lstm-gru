import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import numpy as np
import re

np.random.seed(42)

# 1. INSPECT THE DATA 
test_df = pd.read_csv('sentiment-analysis-dataset/testdata.manual.2009.06.14.csv', encoding='latin1')
train_df = pd.read_csv('sentiment-analysis-dataset/training.1600000.processed.noemoticon.csv', encoding='latin1')

train_df.columns = train_df.columns.str.strip()
print("Shape: ", train_df.shape)
print("Columns: ", train_df.columns)
print("Tweet polarity distribution (%):\n", train_df['polarity of tweet'].value_counts(normalize=True) * 100)
print()
train_df.head(2)

# ⚠️ Test_df is not in the expected, formated. Using train_df as the entire dataset.
# print(f"Test set columns: {test_df.columns}")

# 2. CLEAN THE DATA
def clean_tweet(text: str):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove usernames (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove special characters and numbers (except whitespace)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase and strip leading/trailing whitespace
    return text.lower().strip()

# Apply cleaning function to the tweets
clean_tweets = train_df['text of the tweet'].apply(clean_tweet)
clean_df = train_df.copy()
# Save the cleaned df
clean_df['text of the tweet'] = clean_tweets
clean_df.to_pickle('clean_df.pkl')

# 3. SPLIT DATA: 10% test, 72% train, 18% validation

# Oversample class 4 which is under-represented
# OVERSAMPLING class 4:
    # Tweet polarity distribution (%):
    #  polarity of tweet
    # 0    76.293855
    # 4    23.706145
df = clean_df.copy()

# Define features and target
X = df#['text of the tweet']
y = df['polarity of tweet']

# BALANCED SPLIT: Using OVERSAMPLING to preserve more data
# This approach keeps ALL majority class samples and oversamples the minority class
# Separate data by class
class_0_mask = y == 0
class_4_mask = y == 4

X_class_0 = X[class_0_mask]
y_class_0 = y[class_0_mask]
X_class_4 = X[class_4_mask]
y_class_4 = y[class_4_mask]

print(f"Original - Class 0 samples: {len(X_class_0)}")
print(f"Original - Class 4 samples: {len(X_class_4)}")

# Use ALL samples from class 0 (majority class)
# Oversample class 4 (minority class) to match class 0 size
max_class_size = len(X_class_0)  # Use majority class size as target
print(f"\nTarget size per class: {max_class_size} samples")
print(f"Class 4 needs {max_class_size - len(X_class_4)} additional samples (oversampling)\n")


# For class 0: Use all samples (no sampling needed - preserves all data!)
X_class_0_balanced = X_class_0.copy()
y_class_0_balanced = y_class_0.copy()

# For class 4: Oversample to match class 0 size
# Calculate how many additional samples we need
additional_samples_needed = max_class_size - len(X_class_4)

# Randomly sample WITH replacement from class 4 to create additional samples
oversample_indices = np.random.choice(len(X_class_4), additional_samples_needed, replace=True)

# Get the oversampled data
X_class_4_oversampled = X_class_4.iloc[oversample_indices]
y_class_4_oversampled = y_class_4.iloc[oversample_indices]

# Combine original class 4 samples with oversampled ones
X_class_4_balanced = pd.concat([X_class_4, X_class_4_oversampled], ignore_index=True)
y_class_4_balanced = pd.concat([y_class_4, y_class_4_oversampled], ignore_index=True)

print(f"After oversampling:")
print(f"  Class 0: {len(X_class_0_balanced)} samples (all original samples preserved)")
print(f"  Class 4: {len(X_class_4_balanced)} samples ({len(X_class_4)} original + {additional_samples_needed} oversampled)")
print(f"  Total: {len(X_class_0_balanced) + len(X_class_4_balanced)} samples\n")

# Combine balanced classes
X_balanced = pd.concat([X_class_0_balanced, X_class_4_balanced], ignore_index=True)
y_balanced = pd.concat([y_class_0_balanced, y_class_4_balanced], ignore_index=True)

# Shuffle the combined data
shuffle_indices = np.random.permutation(len(X_balanced))
X_balanced = X_balanced.iloc[shuffle_indices].reset_index(drop=True)
y_balanced = y_balanced.iloc[shuffle_indices].reset_index(drop=True)

# Now split the balanced data: First split to (train+val) and test (90% train+val, 20% test)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.1, random_state=42, stratify=y_balanced
)

# Now split train+val into train and val (80% train, 20% val of the remaining)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)

print(f"Train set shape: {X_train.shape}")
print(f"Val set shape:   {X_val.shape}")
print(f"Test set shape:  {X_test.shape}\n")

def print_class_balance(y, set_name=""):
    value_counts = y.value_counts()
    percentage = value_counts / value_counts.sum() * 100
    balance_df = pd.DataFrame({'count': value_counts, 'percentage': percentage.round(2)})
    print(f"{set_name} class balance:\n{balance_df}\n")

print_class_balance(y_train, "Train")
print_class_balance(y_val, "Val")
print_class_balance(y_test, "Test")

# Save the datasets in .pkl files 
# train
train = X_train.copy()
train['polarity of tweet'] = y_train.values  # keep original column
train['label'] = y_train.values         
# val    
val = X_val.copy()
val['polarity of tweet'] = y_val.values
val['label'] = y_val.values
# test 
test = X_test.copy()
test['polarity of tweet'] = y_test.values
test['label'] = y_test.values

train.to_pickle("train.pkl")
val.to_pickle("val.pkl")
test.to_pickle("test.pkl")