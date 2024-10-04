import argparse
import json
import os.path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pydub import AudioSegment


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.loads(file.read())


def process_media(media_path, segment_name: str, segments: list):
    phrases_to_exclude = ["+BREATH+", "+COUGH+", "+LAUGH+", "+SMACK+", "+AH+", "+EH+", "+MM+", "+GARBAGE+", "+NOISE+"]

    output_folder = os.path.join("output", segment_name)
    create_folder_if_not_exists(output_folder)

    audio = AudioSegment.from_file(media_path)
    metadata = []

    for index, segment in enumerate(segments):
        sequence_number = "{:06}".format(index + 1)

        start_time = segment.get("beg")
        end_time = segment.get("end")
        length = segment.get("len")

        phrase = segment.get("val").strip()
        audio_name = f"{segment_name}_{sequence_number}.mp3"

        if phrase not in phrases_to_exclude and length >= 1000:
            audio_segment = audio[start_time:end_time]
            audio_segment.export(os.path.join(output_folder, audio_name), format="mp3",
                                 codec="libmp3lame", bitrate="320k")

            print(f"Exporting: {audio_name}")

            metadata.append({"file_name": audio_name, "sentence": phrase, "language": "lt"})

    pd.DataFrame(metadata).to_csv(os.path.join(output_folder, "metadata.csv"), index=False)


def main(args: argparse.Namespace):
    create_folder_if_not_exists("output")

    corpus_data = read_json_file(os.path.join(args.input, "etc", "corpus-data.json"))

    tasks: [(str, str, list)] = []

    for key, value in corpus_data.items():
        media_path = os.path.join(args.input, value.get("media").get("path"))
        segment_name = os.path.splitext(os.path.basename(media_path))[0]

        tiers = value.get("tiers", None)
        if tiers is None:
            tasks.append((media_path, segment_name, value.get("speech", [])))
        else:
            for tier_name in tiers:
                segments = value.get("speech").get(tier_name)

                if isinstance(segments, dict):
                    segments = [segments]

                tasks.append((media_path, tier_name, segments))

    with ThreadPoolExecutor(max_workers=12) as executor:
        for task in tasks:
            executor.submit(process_media, task[0], task[1], task[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", help="input directory", type=str, default=None
    )
    arguments = parser.parse_args()
    main(arguments)
