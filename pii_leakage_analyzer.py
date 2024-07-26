from presidio_analyzer import AnalyzerEngine
from datasets import load_dataset
import argparse
import json
import os


def main():
    analyzer = AnalyzerEngine()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--leak_file', type=str, required=True)
    arg_parser.add_argument('--data', type=str, required=True, help="Dataset to analyze, maia-doc, maia-sen, maia-sen-only, bsd-doc, bsd-sen, bsd-sen-only")

    args = arg_parser.parse_args()

    base_data = []

    if args.data == 'maia-doc':
        dataset = load_dataset("data/attack-MAIA-doc-speaker.py", "de-en", split='train')
    elif args.data == 'maia-sen':
        dataset = load_dataset("data/attack-MAIA-sen-speaker.py", "de-en", split='train')
    elif args.data == 'maia-sen-only':
        dataset = load_dataset("data/attack-MAIA-sen-speaker_only.py", "de-en", split='train')
    elif args.data == 'bsd-doc':
        dataset = load_dataset("data/attack_bsd_doc_speaker.py", "de-en", split='train')
    elif args.data == 'bsd-sen':
        dataset = load_dataset("data/attack_bsd_sen_speaker.py", "de-en", split='train')
    elif args.data == 'bsd-sen-only':
        dataset = load_dataset("data/attack_bsd_sen_speaker_only.py", "de-en", split='train')
    else:
        raise Exception("Invalid data type")

    """
    if os.path.exists(f"data/base_{args.data}.text"):
        with open(f"data/base_{args.data}.text", 'r') as f:
            for line in f:
                if line.strip() != "":
                    print(line)
                    base_data.append((line.split("\t")[0], line.split("\t")[1]))
    else:
    """
    for i in range(len(dataset)):
        text = dataset[i]['translation']['en']
        results = analyzer.analyze(
            text=text,
            language='en')
        for result in results:
            base_data.append((text[result.start:result.end], result.entity_type))

    base_data = list(set(base_data))

    with open(f"data/base_{args.data}.text", 'w') as f:
        for item in base_data:
            f.write(f"{item[0]}\t{item[1]}\n")

    leaked_data = []

    with open(args.leak_file, 'r') as f:
        data = json.load(f)
        for example in data["leaked"]:
            results = analyzer.analyze(
                text=example,
                language='en')
            for result in results:
                leaked_data.append((example[result.start:result.end], result.entity_type))

    leaked_data = list(set(leaked_data))

    path = args.leak_file.split("/")
    path = "/".join(path[:-1])

    with open(f"{path}/leaked_{args.data}.text", 'w') as f:
        for item in leaked_data:
            f.write(f"{item[0]}\t{item[1]}\n")

    print("Base data: ", len(base_data))
    print("Leaked data: ", len(leaked_data))
    print("Percentage: ", len(leaked_data)/len(base_data))


if __name__ == '__main__':
    main()