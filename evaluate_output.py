from evaluate import load
from tqdm import tqdm
import argparse
import logging
import glob
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data', type=str, nargs="+", required=True)
    arg_parser.add_argument('--lang', type=str, default='en')
    args = arg_parser.parse_args()

    for data_path in tqdm(args.data, desc="Evaluating..."):
        eval_files = [file for file in glob.glob(f'{data_path}/*.json')
                      if "final_step" in file and not "bertscore" in file]
        for data_name in eval_files:
            with open(data_name, 'r') as f:
                data = json.load(f)

            bertscore = load("bertscore")

            references = data['references']
            predictions = data['predictions']

            results = bertscore.compute(
                predictions=predictions,
                references=references,
                lang=args.lang,
                model_type="allenai/longformer-base-4096",
                rescale_with_baseline=True,
                verbose=True
            )

            f1 = sum(results['f1']) / len(results['f1'])

            data.update({'bertscore_f1': f1})

            logger.info(f"BERTScore F1: {f1}")
            data_name = data_name.split('.')[0]
            with open(f"{data_name}_bertscore.json", 'w') as f:
                json.dump(data, f, indent=4, separators=(',', ': '))

if __name__ == '__main__':
    main()
