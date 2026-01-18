import re
import html
import numpy as np
from difflib import SequenceMatcher

def normalize_title(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s)                 # &amp; -> &, &quot; -> ", &frac12; -> ½, etc.
    s = s.strip()

    # Remove surrounding quotes/brackets repeatedly
    while len(s) >= 2 and ((s[0] == s[-1] and s[0] in "\"'") or (s[0], s[-1]) in [("“","”"), ("‘","’")]):
        s = s[1:-1].strip()

    # Remove stray unmatched leading/trailing quotes
    s = s.strip("\"'“”‘’")

    s = s.lower()

    # Normalize common punctuation variants
    s = s.replace("–", "-").replace("—", "-")

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # Remove spaces around punctuation like commas/hyphens/slashes
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*/\s*", "/", s)

    # Optionally remove most punctuation (uncomment if you want more aggressive matching)
    # s = re.sub(r"[^\w\s/+-]", "", s)

    return s.strip()

def extract_value(line: str, prefix: str) -> str:
    # Get everything after prefix, strip, and try to pull quoted content if present
    val = line[len(prefix):].strip()

    # If it contains a JSON-ish pattern like '"item title" : "..."'
    val = val.replace('"item title" :', '').replace('"Item Title" :', '').strip()

    # If there are quotes, prefer content between the first and last quote
    if '"' in val:
        q = [i for i, c in enumerate(val) if c == '"']
        if len(q) >= 2:
            val = val[q[0] + 1 : q[-1]]
        else:
            val = val.replace('"', '')

    return val.strip()

def get_answers_predictions(file_path):
    answers = []
    preds = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Answer:"):
                answers.append(normalize_title(extract_value(line, "Answer:")))
            elif line.startswith("Generated:"):
                preds.append(normalize_title(extract_value(line, "Generated:")))
            elif line.startswith("LLM:"):  # keep compatibility if you have other files
                preds.append(normalize_title(extract_value(line, "LLM:")))

    return answers, preds

def fuzzy_match(a: str, b: str, threshold: float = 0.93) -> bool:
    # SequenceMatcher ratio in [0,1]; threshold 0.93~0.97 catches small typos but not different items
    return SequenceMatcher(None, a, b).ratio() >= threshold

def evaluate(answers, preds, fuzzy_threshold=None):
    assert len(answers) == len(preds), f"Mismatch: {len(answers)} answers vs {len(preds)} preds"

    n = len(answers)
    exact_hits = 0
    fuzzy_hits = 0

    # Quick debug preview
    print("\nFirst 10 comparisons:")
    for i, (a, p) in enumerate(zip(answers[:10], preds[:10]), 1):
        print(f"{i:02d}. exact={a == p} | answer='{a}' | pred='{p}'")

    for a, p in zip(answers, preds):
        if a == p:
            exact_hits += 1
        if fuzzy_threshold is not None and fuzzy_match(a, p, fuzzy_threshold):
            fuzzy_hits += 1

    exact_acc = exact_hits / n
    results = {
        "count": n,
        "exact_hit@1": exact_acc,
        "exact_ndcg@1": exact_acc,  # same as hit@1 for top-1
    }

    if fuzzy_threshold is not None:
        fuzzy_acc = fuzzy_hits / n
        results.update({
            f"fuzzy_hit@1(th={fuzzy_threshold})": fuzzy_acc,
            f"fuzzy_ndcg@1(th={fuzzy_threshold})": fuzzy_acc,
        })

    return results

if __name__ == "__main__":
    path = "/home/kavach/Dev/Extension-Paper/A-LLMRec/recommendation_output_smolvlm2_clip_lion_G_seed_1.txt"
    answers, preds = get_answers_predictions(path)

    print(f"Loaded answers={len(answers)} preds={len(preds)}")

    # Exact (strict string match)
    res_exact = evaluate(answers, preds, fuzzy_threshold=None)
    print("\nFinal (Exact):")
    for k, v in res_exact.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Fuzzy @ 0.90 (typo-tolerant)
    res_fuzzy = evaluate(answers, preds, fuzzy_threshold=0.90)
    print("\nFinal (Fuzzy, threshold = 0.90):")
    for k, v in res_fuzzy.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


