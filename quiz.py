#!/usr/bin/env python3
"""Exam-style quiz runner for the NCP-GENL labs (standard library only).

    python quiz.py                 # menu: pick a domain
    python quiz.py 05              # practice domain 05 (all of its questions)
    python quiz.py 05 07 -n 10     # 10 random questions from domains 05 and 07
    python quiz.py --review        # only questions you have missed before
    python quiz.py --mock          # 65-question timed mock exam, weighted like the real blueprint
    python quiz.py --stats         # accuracy per domain

Answer with a letter ("B") or, for "Select TWO" questions, several letters ("A,C" or "AC").
Type "q" to stop early; you still get a score report.
"""

import argparse
import json
import random
import sys
import textwrap
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 (Ubuntu 22.04 GPU images)
    import tomli as tomllib
from pathlib import Path

ROOT = Path(__file__).parent
HISTORY = ROOT / ".quiz_history.json"
MOCK_SIZE = 65
MOCK_MINUTES = 120
WIDTH = 92

BOLD, DIM, GREEN, RED, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    BOLD = DIM = GREEN = RED = YELLOW = RESET = ""


def load_bank():
    domains = {}
    for path in sorted(ROOT.glob("labs/*/*/quiz.toml")):
        data = tomllib.loads(path.read_text())
        for q in data["questions"]:
            q["domain"] = data["code"]
        domains[data["code"]] = data
    return domains


def load_history():
    try:
        return json.loads(HISTORY.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_history(history):
    HISTORY.write_text(json.dumps(history, indent=1, sort_keys=True))


def wrap(text, indent=""):
    """Reflow prose to WIDTH but print ``` fenced blocks and list items verbatim."""
    out, para, in_code = [], [], False

    def flush():
        if para:
            out.append(textwrap.fill(" ".join(para), WIDTH, initial_indent=indent, subsequent_indent=indent))
            para.clear()

    for line in textwrap.dedent(text).strip().splitlines():
        if line.strip().startswith("```"):
            flush()
            in_code = not in_code
        elif in_code:
            out.append(indent + "    " + line)
        elif not line.strip():
            flush()
            out.append("")
        elif line.lstrip().startswith(("- ", "* ")):
            flush()
            out.append(indent + line.strip())
        else:
            para.append(line.strip())
    flush()
    return "\n".join(out)


def shuffled_view(q, shuffle):
    """Return (options, correct_letters) with options optionally shuffled."""
    order = list(range(len(q["options"])))
    if shuffle and not q.get("fixed_order", False):
        random.shuffle(order)
    letters = "ABCDEFGH"
    original_correct = {letters.index(a) for a in q["answer"]}
    options = [q["options"][i] for i in order]
    correct = {letters[pos] for pos, i in enumerate(order) if i in original_correct}
    return options, correct


def parse_answer(raw, n_options):
    letters = {c for c in raw.upper() if c.isalpha()}
    valid = set("ABCDEFGH"[:n_options])
    return letters if letters and letters <= valid else None


def ask(q, index, total, shuffle, feedback, deadline=None):
    options, correct = shuffled_view(q, shuffle)
    header = f"[{index}/{total}]  domain {q['domain']} · {q.get('topic', '')}"
    if deadline:
        header += f"  ·  {max(0, int(deadline - time.time()) // 60)} min left"
    print(f"\n{DIM}{header}{RESET}")
    print(BOLD + wrap(q["question"]) + RESET)
    if len(correct) > 1 and "select" not in q["question"].lower():
        print(f"{YELLOW}(Select {len(correct)}.){RESET}")
    for letter, opt in zip("ABCDEFGH", options):
        print(wrap(opt, indent="     ").replace("     ", f"  {letter}) ", 1))

    while True:
        raw = input(f"{DIM}answer>{RESET} ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        chosen = parse_answer(raw, len(options))
        if chosen:
            break
        print(f"Enter letter(s) A–{'ABCDEFGH'[len(options) - 1]}, or q to quit.")

    ok = chosen == correct
    if feedback:
        verdict = f"{GREEN}✔ Correct{RESET}" if ok else f"{RED}✘ Incorrect{RESET} — answer: {','.join(sorted(correct))}"
        print(verdict)
        print(DIM + wrap(q["explanation"], indent="  ") + RESET)
    return ok


def run(questions, shuffle=True, feedback=True, minutes=None):
    history = load_history()
    results = []
    deadline = time.time() + minutes * 60 if minutes else None
    for i, q in enumerate(questions, 1):
        if deadline and time.time() > deadline:
            print(f"\n{RED}Time is up.{RESET}")
            break
        ok = ask(q, i, len(questions), shuffle, feedback, deadline)
        if ok is None:
            break
        results.append((q, ok))
        h = history.setdefault(q["id"], {"seen": 0, "correct": 0})
        h["seen"] += 1
        h["correct"] += int(ok)
        h["last"] = ok
        save_history(history)
    report(results, show_missed=not feedback)


def report(results, show_missed):
    if not results:
        return
    correct = sum(ok for _, ok in results)
    pct = 100 * correct / len(results)
    colour = GREEN if pct >= 70 else YELLOW if pct >= 55 else RED
    print(f"\n{BOLD}Score: {colour}{correct}/{len(results)} ({pct:.0f}%){RESET}")
    by_domain = {}
    for q, ok in results:
        d = by_domain.setdefault(q["domain"], [0, 0])
        d[0] += ok
        d[1] += 1
    if len(by_domain) > 1:
        for code, (c, n) in sorted(by_domain.items()):
            print(f"  domain {code}: {c}/{n}")
    if show_missed:
        missed = [q for q, ok in results if not ok]
        if missed:
            print(f"\n{BOLD}Review the ones you missed:{RESET}")
            for q in missed:
                print(f"\n{DIM}{q['id']} · {q.get('topic', '')}{RESET}")
                print(wrap(q["question"], indent="  "))
                print(GREEN + wrap("Answer: " + " | ".join(q["options"]["ABCDEFGH".index(a)] for a in q["answer"]), indent="  ") + RESET)
                print(DIM + wrap(q["explanation"], indent="  ") + RESET)


def mock_exam(domains):
    """Sample questions in proportion to the official blueprint weights."""
    total_weight = sum(d["weight"] for d in domains.values())
    picked = []
    for d in domains.values():
        k = min(len(d["questions"]), round(MOCK_SIZE * d["weight"] / total_weight))
        picked += random.sample(d["questions"], k)
    random.shuffle(picked)
    return picked


def stats(domains):
    history = load_history()
    print(f"{BOLD}{'code':<5}{'domain':<42}{'weight':>7}{'bank':>6}{'seen':>6}{'acc':>7}{RESET}")
    for code, d in domains.items():
        seen = [history[q["id"]] for q in d["questions"] if q["id"] in history]
        n_seen = sum(h["seen"] for h in seen)
        acc = f"{100 * sum(h['correct'] for h in seen) / n_seen:.0f}%" if n_seen else "-"
        print(f"{code:<5}{d['domain']:<42}{d['weight']:>6}%{len(d['questions']):>6}{len(seen):>6}{acc:>7}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domains", nargs="*", help="domain codes, e.g. 01 05")
    ap.add_argument("-n", type=int, help="number of questions")
    ap.add_argument("--mock", action="store_true", help=f"{MOCK_SIZE}-question timed mock exam")
    ap.add_argument("--review", action="store_true", help="only questions you got wrong last time")
    ap.add_argument("--stats", action="store_true", help="show accuracy per domain")
    ap.add_argument("--no-shuffle", action="store_true", help="keep answer options in authored order")
    args = ap.parse_args()

    domains = load_bank()
    if args.stats:
        return stats(domains)

    if args.mock:
        print(f"{BOLD}Mock exam:{RESET} {MOCK_SIZE} questions, {MOCK_MINUTES} minutes, no feedback until the end.")
        return run(mock_exam(domains), shuffle=not args.no_shuffle, feedback=False, minutes=MOCK_MINUTES)

    codes = args.domains
    if not codes and not args.review:
        stats(domains)
        codes = input("\nDomain code(s) to practise (blank = all): ").split()
    codes = codes or list(domains)
    unknown = [c for c in codes if c not in domains]
    if unknown:
        sys.exit(f"Unknown domain(s): {', '.join(unknown)}. Choose from {', '.join(domains)}.")

    pool = [q for c in codes for q in domains[c]["questions"]]
    if args.review:
        history = load_history()
        pool = [q for q in pool if history.get(q["id"], {}).get("last") is False]
        if not pool:
            sys.exit("Nothing to review — no missed questions in those domains.")
    random.shuffle(pool)
    run(pool[: args.n] if args.n else pool, shuffle=not args.no_shuffle)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
