"""Day 4: measure retrieval quality (Hit@1, Hit@K) and refusal on out-of-corpus questions."""
import json
import sys
import time

import config
from agent import agent_answer as answer
from ask import retrieve

REFUSAL_MARKERS = ("je ne sais pas", "i don't know", "i do not know", "ne sais pas")


def as_list(expected):
    return expected if isinstance(expected, list) else [expected]


def main(path="eval_questions.json"):
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)

    hit1 = hitk = n_in = 0
    refused = n_out = 0
    rows = []

    for item in questions:
        q, expected = item["question"], item["expected"]

        if expected is None:
            n_out += 1
            text, _ = answer(q)
            ok = any(m in text.lower() for m in REFUSAL_MARKERS)
            refused += ok
            rows.append((q, "(hors corpus)", "refus OK" if ok else "A INVENTE", ""))
            time.sleep(15)
        else:
            n_in += 1
            hits = retrieve(q)
            files = [m["source"] for _, m, _ in hits]
            exp = as_list(expected)
            rank = next((i + 1 for i, f in enumerate(files) if f in exp), None)
            hit1 += rank == 1
            hitk += rank is not None
            rows.append((q, ",".join(exp), f"rang {rank}" if rank else "ABSENT", files[0]))

    print(f"\nCHUNK_WORDS = {getattr(config, 'CHUNK_WORDS', '?')}  |  TOP_K = {config.TOP_K}\n")
    for q, exp, res, top in rows:
        print(f"- {q[:55]:55} | attendu: {exp:32} | {res:10} | n°1: {top}")

    print("\n=== Résultats ===")
    if n_in:
        print(f"Hit@1 : {hit1}/{n_in} ({100 * hit1 / n_in:.0f} %)")
        print(f"Hit@{config.TOP_K} : {hitk}/{n_in} ({100 * hitk / n_in:.0f} %)")
    if n_out:
        print(f"Refus correct (hors corpus) : {refused}/{n_out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
