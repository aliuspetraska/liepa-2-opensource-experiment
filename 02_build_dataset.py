import os

from datasets import load_dataset, concatenate_datasets, Audio
from huggingface_hub import login


def main():
    login(token="<put your token here>")

    datasets = []

    for subdirectory in os.listdir("output"):
        subdirectory_path = os.path.join("output", subdirectory)
        if os.path.isdir(subdirectory_path):
            try:
                dataset = load_dataset(subdirectory_path)
                datasets.append(dataset["train"])
            except Exception as e:
                print(e)

    dataset = concatenate_datasets(datasets)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, mono=True, decode=True))

    liepa_voice = dataset.train_test_split(test_size=0.05, shuffle=True)

    liepa_voice.save_to_disk("dataset")

    liepa_voice.push_to_hub(private=True, repo_id="<put your repo id here>")


if __name__ == "__main__":
    main()
