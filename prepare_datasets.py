from datasets import load_dataset
import json

TARGET_SAMPLES = 1000

# -----------------------
# GSM8K
# -----------------------
gsm8k = load_dataset("gsm8k", "main", split=f"train[:{TARGET_SAMPLES}]")

with open("data/gsm8k_subset.jsonl", "w", encoding="utf-8") as f:
    for i, sample in enumerate(gsm8k):
        answer = sample["answer"].split("####")[-1].strip()
        row = {
            "id": f"gsm8k_{i+1}",
            "dataset": "gsm8k",
            "context": "",
            "question": sample["question"],
            "answer": answer
        }
        f.write(json.dumps(row) + "\n")


# -----------------------
# SQuAD v2
# -----------------------
squad = load_dataset("squad_v2", split="train")

with open("data/squad_v2_subset.jsonl", "w", encoding="utf-8") as f:
    written = 0
    for i, sample in enumerate(squad):
        if sample["answers"]["text"]:
            answer = sample["answers"]["text"][0]
        else:
            continue

        if written >= TARGET_SAMPLES:
            break

        row = {
            "id": f"squad_{written+1}",
            "dataset": "squad_v2",
            "context": sample["context"],
            "question": sample["question"],
            "answer": answer
        }
        f.write(json.dumps(row) + "\n")
        written += 1


# -----------------------
# HotpotQA
# -----------------------
hotpot = load_dataset("hotpot_qa", "distractor", split=f"validation[:{TARGET_SAMPLES}]")

with open("data/hotpotqa_subset.jsonl", "w", encoding="utf-8") as f:

    for idx, row in enumerate(hotpot):

        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]

        context = ""

        for title, sent_list in zip(titles, sentences):
            paragraph = " ".join(sent_list)
            context += f"{title}: {paragraph}\n\n"

        sample = {
            "id": f"hotpot_{idx+1}",
            "dataset": "hotpot_qa",
            "context": context,
            "question": row["question"],
            "answer": row["answer"]
        }

        f.write(json.dumps(sample) + "\n")

# -----------------------
# TriviaQA
# -----------------------

trivia = load_dataset("trivia_qa", "rc", split="train", streaming=True)

with open("data/triviaqa_subset.jsonl", "w", encoding="utf-8") as f:
    for i, sample in enumerate(trivia):

        if i >= TARGET_SAMPLES:
            break

        context = sample["entity_pages"]["wiki_context"][0] if sample["entity_pages"]["wiki_context"] else ""
        answer = sample["answer"]["value"]

        row = {
            "id": f"trivia_{i+1}",
            "dataset": "trivia_qa",
            "context": context,
            "question": sample["question"],
            "answer": answer
        }

        f.write(json.dumps(row) + "\n")