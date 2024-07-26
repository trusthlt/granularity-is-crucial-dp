# Granularity is crucial when applying differential privacy to text: An investigation for neural machine translation

## Description

Code accompanying paper "Granularity is crucial when applying differential privacy to text: An investigation for neural machine translation".
We investigate training NMT at both the sentence and document levels with [differentially private stochastic gradient descent (DP-SGD)](https://arxiv.org/abs/1607.00133), analyzing the privacy/utility trade-off for both scenarios, and evaluating the risks of not using the appropriate privacy granularity in terms of leaking personally identifiable information (PII).
The code is a fork of the [DP-NMT](https://github.com/trusthlt/dp-nmt) framework, which includes a [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax) implementation of the DP-SGD algorithm for training neural machine translation models with differential privacy.

## Installation 

With Miniconda installed, create a new environment with the required dependencies:

```bash
conda create -n dp-nmt python=3.11
conda activate dp-nmt
git clone URL
cd URL
pip install -r requirements.txt
conda install cuda-nvcc -c conda-forge -c nvidia
```
## Available Datasets

We provided 2 processed datasets for the experiments: BSD [(Rikters et al. 2019)](https://aclanthology.org/D19-5204/), MAIA [(Farinha et al., 2022)](https://aclanthology.org/2022.wmt-1.70/). Those dataset are available in both sentence and document level. In both datasets, we concatenate the utterances within a dialogue into a single document for document-level machine translation. For example, we concatenate the speaker’s name and the utterance into a single sentence. Namely, `<SPEAKER>: <UTTERANCE>`. Then we concatenate all the utterances within a dialogue into a single document to form document-level training pairs.

### BSD processed examples

The dataset is a collection of fictional business conversations in various scenarios (e.g. "face-to-face", "phone call", "meeting"), with parallel data for Japanese and English.

Sentence-level original training pairs provided by the dataset
```
...
{
"ja_speaker": 土井さん
"ja_sentence": 稲田さん、H社の高市様からお電話です。
"en_speaker": Doi-san
"en_sentence": Inada-san, you have a call from Mr. Takaichi of Company H.
}
{
"ja_speaker": 稲田さん
"ja_sentence": もしもし、稲田です。
"en_speaker": Inada-san
"en_sentence": Hello, this is Inada.
}
...
```

Our processed sentence-level training pairs
```
{
...
"ja": 土井さん: 稲田さん、H社の高市様からお電話です。
"en": Doi-san: Inada-san, you have a call from Mr. Takaichi of Company H.
"ja": 稲田さん: もしもし、稲田です。
"en": Inada-san: Hello, this is Inada.
...
}
```
Our processed document-level training pair (utterances within a dialogue)
```
{
"ja": ...  土井さん: 稲田さん、H社の高市様からお電話です。 稲田さん: もしもし、稲田です。...
"en": ... Doi-san: Inada-san, you have a call from Mr. Takaichi of Company H. Inada-san: Hello, this is Inada. ...
}
```

### MAIA processed examples

The Multilingual Artificial Intelligence Agent Assistant (MAIA) corpus consists of genuine bilingual (German-English) customer support.

Sentence-level original training pairs provided by the dataset

```
{
...
"de": Hallo, können sie mir sagen wann das bestellte Bett ca Versand wird?
"en": Hello, Can you tell me when the ordered bed will be approximately shipped?
"de": Guten Morgen #NAME#,
"en": Good Morning #NAME#.
"de": vielen Dank, dass Sie #PRS_ORG# kontaktiert haben. Ich hoffe, dass es Ihnen gut geht.
"en": Thank you for contacting #PRS_ORG# I hope you are well.
...
}
```
Our processed sentence-level dataset with artificially replaced PII training pairs
```
{
...
"de": Kunde: Hallo, können sie mir sagen wann das bestellte Bett ca Versand wird?
"en": Customer: Hello, Can you tell me when the ordered bed will be approximately shipped?
"de": Agent: Guten Morgen Olav Kusch,
"en": Agent: Good Morning Olav Kusch.
"de": Agent: vielen Dank, dass Sie Hethur Ullmann GmbH & Co. KG kontaktiert haben. Ich hoffe, dass es Ihnen gut geht.
"en": Agent: Thank you for contacting Hethur Ullmann GmbH & Co. KG I hope you are well.
...
}
```

Our processed document-level dataset with artificially replaced PII training pairs (utterances within a dialogue)

```
{
"de": ... Kunde: Hallo, können sie mir sagen wann das bestellte Bett ca Versand wird? Agent: Guten Morgen Olav Kusch Agent: vielen Dank, dass Sie Hethur Ullmann GmbH & Co. KG kontaktiert haben. Ich hoffe, dass es Ihnen gut geht. ...
"en": ... Customer: Hello, Can you tell me when the ordered bed will be approximately shipped? Agent: Good Morning Olav Kusch. Agent: Thank you for contacting Hethur Ullmann GmbH & Co. KG I hope you are well. ...
}
```
## Running Experiments

The following commands show how to run the experiments for both sentence and document levels with and without differential privacy. The sentence and document level hyperparameters are different, mostly due to the input and output sequence lengths. Also the batch size is different, as the document-level training requires more memory, typically a smaller batch size is used (1 or 2).

### Normal runs

#### Sentence level

```bash
python main.py \
  --dataset data/bsd_sen_speaker.py \
  --model agemagician/mlong-t5-tglobal-base \
  --lang_pair ja-en \
  --epochs 15 \
  --train_batch_size 16 \
  --input_max_seq_len 64 \
  --output_max_seq_len 64 \
  --learning_rate 0.0001 \
  --eval_batch_size 16 \
  --early_stopping False
```

#### Document level

```bash
python main.py \
  --dataset data/bsd_doc_speaker.py \
  --model agemagician/mlong-t5-tglobal-base \
  --lang_pair ja-en \
  --epochs 25 \
  --train_batch_size 1 \
  --input_max_seq_len 1030 \
  --output_max_seq_len 1090 \
  --learning_rate 0.0001 \
  --eval_batch_size 4 \
  --early_stopping False
```

### DP-SGD private runs

To determine the noise multiplier, you need to use the `compute_epsilon.py` script.

#### Sentence level

```bash
python main.py \
  --dataset data/bsd_sen_speaker.py \
  --model agemagician/mlong-t5-tglobal-base \
  --lang_pair ja-en \
  --epochs 15 \
  --train_batch_size 8 \
  --input_max_seq_len 64 \
  --output_max_seq_len 64 \
  --learning_rate 0.001 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --private True \
  --sampling_method poisson_sampling \
  --noise_multiplier 0.5000116530059 \
```
#### Document level

```bash
python main.py \
  --dataset data/bsd_doc_speaker.py \
  --model agemagician/mlong-t5-tglobal-base \
  --lang_pair ja-en \
  --epochs 50 \
  --train_batch_size 2 \
  --input_max_seq_len 1030 \
  --output_max_seq_len 1090 \
  --learning_rate 0.001 \
  --eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --private True \
  --sampling_method poisson_sampling \
  --noise_multiplier 0.263657997102083 \
```

### Data extraction attacks

In this attack, we first sample the sentence-level training data such that they have the same size as the number of validation and test data. Then we forward all sampled data through the target model (e.g. private training) and extract the logits for the loss. This is also done for the validation and test data. Finally, we compare the average loss of the training data via the base model (e.g. normal training) with the loss of the validation, test, and the sampled data of the target model. If the True Positive Rate (TPR) minus False Positive Rate (FPR) is greater than 0.5, we consider the model as vulnerable to data extraction attacks.

```bash
python main_attack.py \
  --seed 666 \
  --dataset_infer data/attack-MAIA-sen-speaker_only.py \
  --dataset_loss data/MAIA-sen-speaker.py \
  --lang_pair de-en \
  --base_model maia-sen-speaker-de-en-final/2023_10_22-09_13_19/mlong-t5-tglobal-base \
  --target_model checkpoints/2023_12_29-21_41_29501926/mlong-t5-tglobal-base \
  --batch_size 16 \
  --input_max_seq_len 128 \
  --output_max_seq_len 128
```
