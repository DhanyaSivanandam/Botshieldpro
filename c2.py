import pandas as pd
import os

BASE_PATH = "cresci-2017"

folders_with_labels = {
    "genuine_accounts": 1,
    "fake_followers": 0,
    "social_spambots_1": 0,
    "social_spambots_2": 0,
    "social_spambots_3": 0,
    "traditional_spambots_1": 0,
    "traditional_spambots_2": 0,
    "traditional_spambots_3": 0,
    "traditional_spambots_4": 0
}

all_data = []

for folder_name, label in folders_with_labels.items():

    folder_path = os.path.join(BASE_PATH, folder_name)

    if not os.path.isdir(folder_path):
        print("Missing folder:", folder_name)
        continue

    users_file = os.path.join(folder_path, "users.csv")

    if not os.path.isfile(users_file):
        print("Missing users.csv in:", folder_name)
        continue

    print("Loading:", users_file)

    df = pd.read_csv(users_file)
    df["label"] = label
    all_data.append(df)

if len(all_data) == 0:
    raise Exception("No datasets found.")

final_df = pd.concat(all_data, ignore_index=True)

final_df.to_csv("final_dataset.csv", index=False)

print("Dataset prepared successfully!")
