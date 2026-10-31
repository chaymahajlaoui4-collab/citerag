"""Day 5: agent layer. The LLM decides when to call search_docs, up to MAX_ROUNDS times."""
import json
import sys

import config
from ask import retrieve, _llm, SYSTEM_PROMPT

import time
import groq


def call_llm(messages, **kwargs):
    """Call Groq, retrying after a pause when the per-minute token limit is hit."""
    for attempt in range(4):
        try:
            return _llm.chat.completions.create(
                model=config.LLM_MODEL, temperature=0, messages=messages, **kwargs
            )
        except (groq.RateLimitError, groq.APIStatusError) as e:
            if getattr(e, "status_code", None) not in (413, 429) or attempt == 3:
                raise
            print(f"(limite Groq atteinte, pause de 20 s, essai {attempt + 2}/4)")
            time.sleep(20)


MAX_ROUNDS = 3

AGENT_PROMPT = (
    "You are a technical documentation assistant with a tool search_docs(query). "
    "Use the tool for any question about the documentation. You may call it several "
    "times with different queries, for example one per concept in a comparison, or a "
    "rephrased query in English if the first results are off-topic. "
    "Answer ONLY using facts explicitly stated in the tool results. Do NOT use outside "
    "knowledge. A link or a reference to another page is NOT an answer. "
    "Cite every claim as [source, p.X]. If the results do not contain the answer, "
    "reply exactly: \"Je ne sais pas : l'information n'est pas dans les documents.\" "
    "(in the language of the question). For greetings or small talk, answer briefly "
    "without searching. Reply in the same language as the question."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the indexed Camunda documentation and return the most relevant passages.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}]


def search_docs(query):
    hits = retrieve(query)
    text = "\n\n".join(f"[{m['source']}, p.{m['page']}]\n{d[:1200]}" for d, m, _ in hits)
    return text, hits


def final_answer(question, all_hits, queries):
    """Last step: plain call without tool history, using every passage found so far."""
    seen, parts = set(), []
    for d, m, _ in all_hits:
        key = (m["source"], d)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"[{m['source']}, p.{m['page']}]\n{d[:1200]}")
    context = "\n\n".join(parts[:8])
    resp = call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ])
    return resp.choices[0].message.content or "", all_hits, queries


def run_agent(question):
    messages = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": question},
    ]
    all_hits, queries = [], []

    for round_no in range(MAX_ROUNDS + 1):
        # last round: tools disabled, the model must answer with what it has
        if round_no == MAX_ROUNDS:
            return final_answer(question, all_hits, queries)
        resp = call_llm(messages, tools=TOOLS, tool_choice="auto")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "", all_hits, queries

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                query = json.loads(tc.function.arguments)["query"]
                result, hits = search_docs(query)
                all_hits += hits
                queries.append(query)
            except Exception as e:  # bad arguments or unknown tool: tell the model
                result = f"Tool error: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "", all_hits, queries


def agent_answer(question):
    """Same return shape as ask.answer: (text, hits)."""
    text, hits, _ = run_agent(question)
    return text, hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Question: ")
    text, hits, queries = run_agent(q)
    print("\n" + text)
    print(f"\n--- Agent: {len(queries)} recherche(s) ---")
    for query in queries:
        print(f"  search_docs({query!r})")
    print("--- Passages vus par l'agent ---")
    for d, m, dist in hits:
        print(f"{m['source']} p.{m['page']} (distance {dist:.3f})")
