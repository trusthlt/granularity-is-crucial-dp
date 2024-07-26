import random
import json

random.seed(2024)

with open("MAIA-doc-speaker/MAIA-de-en-train_doc.json", "r") as f:
    maia_data_doc = json.load(f)

with open("MAIA-sen-speaker/MAIA-de-en-train_sentence.json", "r") as f:
    maia_data_sen = json.load(f)

sampled_maia_data_doc = random.sample(maia_data_doc, 141)
sampled_maia_data_sen = []
for data in sampled_maia_data_doc:
    for sentence in maia_data_sen:
        if sentence["doc_id"] == data["doc_id"]:
            sampled_maia_data_sen.append(sentence)

with open("sampled_maia_data_doc.json", "w") as f:
    json.dump(sampled_maia_data_doc, f, indent=4)

with open("sampled_maia_data_sen.json", "w") as f:
    json.dump(sampled_maia_data_sen, f, indent=4)

with open("bsd_attack/train.json", "r") as f:
    bsd_data = json.load(f)

sampled_bsd_data = random.sample(bsd_data, 138)

with open("sampled_bsd_data.json", "w") as f:
    json.dump(sampled_bsd_data, f, indent=4)


sampled_maia_data_sen_only = random.sample(maia_data_sen, 4597)
sampled_bsd_data_only = []
for data in bsd_data:
    data_renew = []
    for conversation in data["conversation"]:
        one_data = {
            "id": data["id"],
            "tag": data["tag"],
            "title": data["title"],
            "original_language": data["original_language"],
            "conversation": [conversation]
        }
        data_renew.append(one_data)
    sampled_bsd_data_only.extend(data_renew)

sampled_bsd_data_only = random.sample(sampled_bsd_data_only, 4171)

with open("sampled_maia_data_sen_only.json", "w") as f:
    json.dump(sampled_maia_data_sen_only, f, indent=4)

with open("sampled_bsd_data_only.json", "w") as f:
    json.dump(sampled_bsd_data_only, f, indent=4)
