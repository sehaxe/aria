#!/usr/bin/env python3
"""Assemble Aria pretrain mix from HF + local sources into plaintext staging.

Runs in .corpusenv (datasets/pyarrow/huggingface_hub/teich/zstandard).
Writes .txt into data/pretrain/mix/<cat>/ and mix/code/<lang>/.
Pack:  source .venv315t/bin/activate && python prepare_data.py data/pretrain/mix data/pretrain/real/corpus.bin

Gated NVIDIA sets were replaced by open fineweb-family alternatives:
  web      -> HuggingFaceFW/fineweb-edu (premium edu-filtered web)
  code     -> OpenCoder-LLM/opc-fineweb-code-corpus (fineweb-filtered, mixed langs)
Other NVIDIA sets (Nemotron-CC-Math-v1, Agentic-SFT, Instruct-SFT) are open
and used directly. Still-skipped gated sets fall back to the next source.
Text extraction is schema-agnostic: plain 'text', code 'content'/'text', qa
'problem'+'solution', and chat 'messages'/'conversations'.
Per-language code tries data_dir=<lang>, then a detected language column,
then yields the whole repo (mixed). The OPC code repo is mixed ('*'): streamed once.
"""
import argparse, glob, json, os, random, signal, sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset

ROOT = Path("/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain")
MIX = ROOT / "mix"
LOCAL_HF = Path("/home/sehaxe/Downloads/hf")
ARXIV = Path("/home/sehaxe/Downloads/arxiv-ai")
GB = 1024 ** 3
CHUNK_BYTES = 6 * 1024 * 1024


class SourceTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise SourceTimeout()

PRIMARY_TEXT = ("text", "content", "code", "source", "markdown", "utterance", "headline")
QA_Q = ("problem", "question", "prompt", "instruction")
QA_A = ("solution", "answer", "response", "completion", "output")
CHAT = ("messages", "conversation", "conversations", "dialog", "chat")
LANG_COLS = ("language", "lang", "language_name", "programming_language")

# Ideal base-model mix: intellect (code/agentic/math/reasoning) dominates,
# strong bilingual (en web + ru web/dialogs/books), wiki for grounding.
# No SFT chat slop. All sources verified loadable via datasets==5.0 streaming.
CAT_BUDGET = {
    # ponytail: bilingual en+ru, NOT a ru model -> web(en) > ruweb. ruweb = RU LANGUAGE (vocab/register), not intellect.
    # ponytail: rudialog+chatter dropped from pretrain (old-model slop); their 0.06 -> wiki/books. wiki>books: wiki=dense facts, books=style only.
    # ponytail: fineweb-2 en DROPPED (thinner than edu for 105M); web = edu + dclm. ruweb = fineweb-2 rus_Cyrl.
    "code": 0.29, "agentic": 0.22, "math": 0.11, "web": 0.13, "ruweb": 0.05,
    "wiki": 0.10, "books": 0.05, "reasoning": 0.04,
    "synthetic": 0.005, "qa": 0.005,
}

CODE_LANGS = {"*": 1.0}  # OPC fineweb-code is mixed (all langs); streamed once

# Ordered per category: agentic FIRST (so auto_train trigger fires fast), then
# grounding (wiki/books/qa/synthetic), already-full cats (web/ruweb/reasoning/code/math)
# LAST — build is resume-aware (cat_bytes skips already-downloaded), so they're no-ops.
# All sources verified loadable via datasets==5.0 streaming (2026-07-17).
SOURCES = [
    # --- AGENTIC (tool-use + trajectories, verified, fresh 2025-2026) ---
    {"cat": "agentic", "kind": "hf", "repo": "nvidia/Nemotron-SFT-Agentic-v2", "split": "interactive_agent"},
    {"cat": "agentic", "kind": "hf", "repo": "nvidia/Nemotron-SFT-Agentic-v2", "split": "tool_calling"},
    # ponytail: RL-NVIDIA (Qwen3/DeepSeek-V3 teachers) + Toucan(Kimi-K2) dropped per user: "устарели ужасно"
    {"cat": "agentic", "kind": "hf", "repo": "glaiveai/glaive-function-calling-v2", "split": "train"},
    {"cat": "agentic", "kind": "hf", "repo": "NousResearch/hermes-function-calling-v1", "split": "train"},
    # --- AGENTIC CORE: frontier-teacher distillation (GPT-5.5/Grok-4/Mythos-5/Fable-5) ---
    # ponytail: suayptalha(0.2MB/s) + TerraBytes(1.5MB/s) Fable-5 traces dropped: 2-15x slower, same teacher family. Manusagents = fastest + freshest, dominates budget.
    {"cat": "agentic", "kind": "hf", "repo": "Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset", "split": "train"},
    {"cat": "agentic", "kind": "hf", "repo": "WithinUsAI/claude_mythos_distilled_25k", "split": "train"},
    # --- WIKI / BOOKS (grounding) ---
    {"cat": "wiki", "kind": "hf", "repo": "wikimedia/wikipedia", "split": "train", "config": "20231101.en"},
    {"cat": "wiki", "kind": "hf", "repo": "wikimedia/wikipedia", "split": "train", "config": "20231101.ru"},
    {"cat": "books", "kind": "gutenberg", "repo": "en-classics"},
    {"cat": "books", "kind": "libru", "repo": "ru-classics"},
    # --- QA (light) ---
    {"cat": "qa", "kind": "hf", "repo": "google-research-datasets/natural_questions", "split": "train"},
    # --- SYNTHETIC (math distill) ---
    {"cat": "synthetic", "kind": "hf", "repo": "nvidia/Nemotron-CC-Math-v1", "split": "train", "config": "4plus"},
    # --- WEB (en): dense-intellect (edu) + dclm; already full, resume-skips ---
    {"cat": "web", "kind": "hf", "repo": "HuggingFaceFW/fineweb-edu", "split": "train"},
    {"cat": "web", "kind": "hf", "repo": "mlfoundations/dclm-baseline-1.0", "split": "train", "fallback": True},
    # --- RUWEB (ru): fineweb-2 rus_Cyrl — already full, resume-skips ---
    {"cat": "ruweb", "kind": "hf", "repo": "HuggingFaceFW/fineweb-2", "split": "train", "config": "rus_Cyrl"},
    # --- REASONING (modern CoT, 2026 teacher MIX) — already full, resume-skips ---
    # ponytail: dropped 2024-25 reasoning + Opus-4.6/4.7 + GLM-5.1 (all stale/redundant; Axiom already mixes GLM-5.2/Kimi2.6/Opus4.7/Mythos5/Fable5/Qwen3.7).
    {"cat": "reasoning", "kind": "hf", "repo": "shreyan35/Project-Axiom-1.0-Opus4.7-Kimi2.6-GLM5.2-Deepseek4-Mythos5-Fable5-Qwen3.7", "split": "train"},
    # --- CODE (verified) — already full, resume-skips ---
    {"cat": "code", "kind": "code", "repo": "OpenCoder-LLM/opc-fineweb-code-corpus"},
    {"cat": "code", "kind": "hf", "repo": "HuggingFaceTB/smollm-corpus", "split": "train", "config": "python-edu"},
    {"cat": "code", "kind": "hf", "repo": "KodCode/KodCode-V1", "split": "train"},
    {"cat": "code", "kind": "hf", "repo": "glaiveai/glaive-code-assistant", "split": "train"},
    # --- MATH (LOCAL fresh GLM-5.1/Kimi-K2.5 math only; ~10.3GB via converter) — already full, resume-skips ---
    # ponytail: dropped OpenR1/Nemotron-CC-Math HF (user: 10GB local is enough, don't top-up).
    # Reader handles .jsonl natively (dataset.py _read_docs); no converter needed.
]


def extract_text(row):
    for mf in CHAT:
        v = row.get(mf)
        if v:
            parts = []
            for m in v:
                if isinstance(m, dict):
                    parts.append(str(m.get("content") or m.get("text") or m.get("value") or ""))
                else:
                    parts.append(str(m))
            joined = "\n\n".join(p for p in parts if str(p).strip())
            if joined.strip():
                return joined
    for f in PRIMARY_TEXT:
        v = row.get(f)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list):
            return "\n".join(str(x) for x in v if str(x).strip())
    q = [str(row[f]) for f in QA_Q if row.get(f)]
    a = [str(row[f]) for f in QA_A if row.get(f)]
    if q or a:
        return "\n\n".join(q + a)
    out = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 40]
    return "\n\n".join(out)


def iter_hf(repo, split="train", data_dir=None, config=None):
    kw = {"streaming": True, "split": split}
    if data_dir:
        kw["data_dir"] = data_dir
    if config:
        kw["name"] = config
    yield from load_dataset(repo, **kw)


def iter_local_parquet(dirpath):
    files = sorted(glob.glob(str(Path(dirpath) / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet in {dirpath}")
    for f in files:
        for batch in pq.read_table(f).to_batches():
            for row in batch.to_pylist():
                yield row


def _lang_match(row, lang):
    return any(str(row.get(c) or "").lower() == lang.lower() for c in LANG_COLS)


def iter_code(repo, lang):
    local = LOCAL_HF / repo
    if local.exists():
        for row in iter_local_parquet(local):
            if lang in (None, "*") or _lang_match(row, lang):
                yield row
        return
    if lang in (None, "*"):
        yield from iter_hf(repo)   # mixed code repo (e.g. OPC fineweb-code)
        return
    try:
        yield from iter_hf(repo, data_dir=lang)
        return
    except Exception:
        pass
    ds = iter_hf(repo)
    row = next(ds)
    lcol = next((c for c in LANG_COLS if c in row), None)
    if lcol:
        if _lang_match({lcol: row.get(lcol)}, lang):
            yield row
        for r in ds:
            if _lang_match(r, lang):
                yield r
    else:
        for r in ds:
            yield r


def iter_arxiv():
    seen = set()
    for f in sorted(glob.glob(str(ARXIV / "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                pid = d.get("id")
                if pid in seen:
                    continue
                seen.add(pid)
                title, abstract = d.get("title", ""), d.get("abstract", "")
                authors = ", ".join(d.get("authors", [])[:8])
                cats = ", ".join(d.get("categories", []))
                txt = f"{title}\n\n{abstract}"
                if authors:
                    txt += f"\n\nAuthors: {authors}"
                if cats:
                    txt += f"\nCategories: {cats}"
                yield txt


class Writer:
    def __init__(self, base_dir):
        self.base = base_dir
        self.cur, self.size, self.n, self.fidx = [], 0, 0, 0
        self.src = base_dir.name

    def add(self, text):
        text = text.strip()
        if not text:
            return
        self.cur.append(text)
        self.size += len(text.encode("utf-8")) + 1
        self.n += 1
        if self.size >= CHUNK_BYTES:
            self.flush()

    def flush(self):
        if not self.cur:
            return
        self.base.mkdir(parents=True, exist_ok=True)
        p = self.base / f"{self.src}_{self.fidx:05d}.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(self.cur))
        self.cur, self.size, self.fidx = [], 0, self.fidx + 1


def cat_bytes(cat):
    pat = (MIX / "code") if cat == "code" else (MIX / cat)
    return sum(p.stat().st_size for p in pat.rglob("*.txt"))


GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub"
# Классика в public domain: топ-авторы по ID книг (en). Минимум, но качественно.
GUTENBERG_BOOKS = [
    1342, 11, 1661, 2701, 84, 98, 1952, 280, 1080, 1400, 174, 161, 768, 345,
    4300, 2591, 1232, 2554, 158, 974, 1086, 2852, 2090, 3207, 4300, 1497,
]
LIBRU_AUTHORS = [
    "https://az.lib.ru/t/tolstoy_lew_nikolaevich/",
    "https://az.lib.ru/d/dostoewskij_fedor_mihajlowich/",
    "https://az.lib.ru/c/chehow_anton_pawlowich/",
    "https://az.lib.ru/b/bulgakov_mikhail_afanasevich/",
    "https://az.lib.ru/p/pushkin_aleksandr_sergeevich/",
    "https://az.lib.ru/g/gogol_nikolaj_wasiliewich/",
    "https://az.lib.ru/l/lukin_n/",
    "https://az.lib.ru/m/maykow_ap/",
]


def iter_gutenberg():
    import urllib.request
    for bid in GUTENBERG_BOOKS:
        url = f"{GUTENBERG_MIRROR}/{bid}/pg{bid}.txt"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                txt = r.read().decode("utf-8", "ignore")
            if len(txt) > 1000:
                yield txt
        except Exception:
            continue


def iter_libru():
    import urllib.request, re
    for base in LIBRU_AUTHORS:
        try:
            with urllib.request.urlopen(base, timeout=30) as r:
                html = r.read().decode("utf-8", "ignore")
            links = set(re.findall(r'href="([^"]+\.txt)"', html))
            for link in list(links)[:8]:
                u = link if link.startswith("http") else base + link
                try:
                    with urllib.request.urlopen(u, timeout=30) as r2:
                        txt = r2.read().decode("utf-8", "ignore")
                    if len(txt) > 2000:
                        yield txt
                except Exception:
                    continue
        except Exception:
            continue


def source_rows(src, limit_bytes):
    if limit_bytes <= 0:
        return
    if src["kind"] == "gutenberg":
        w_ = Writer(MIX / "books")
        for txt in iter_gutenberg():
            w_.add(txt)
            if w_.size >= limit_bytes:
                break
        w_.flush()
        return
    if src["kind"] == "libru":
        w_ = Writer(MIX / "books")
        for txt in iter_libru():
            w_.add(txt)
            if w_.size >= limit_bytes:
                break
        w_.flush()
        return
    if src["kind"] == "arxiv":
        w_ = Writer(MIX / "papers")
        for txt in iter_arxiv():
            w_.add(txt)
            if w_.size >= limit_bytes:
                break
        w_.flush()
        return
    if src["kind"] == "code":
        wsum = sum(CODE_LANGS.values())
        for lang, w in CODE_LANGS.items():
            llimit = limit_bytes * w / wsum
            if llimit <= 0:
                continue
            out_dir = MIX / "code" if lang == "*" else MIX / "code" / lang
            w_ = Writer(out_dir)
            try:
                for row in iter_code(src["repo"], lang):
                    w_.add(extract_text(row))
                    if w_.size >= llimit:
                        break
            except Exception as e:
                print(f"  [skip] code {lang} {src['repo']}: {type(e).__name__}: {str(e)[:90]}")
            w_.flush()
        return
    repo, split = src["repo"], src.get("split", "train")
    try:
        if (LOCAL_HF / repo).exists():
            rows = iter_local_parquet(LOCAL_HF / repo)
        else:
            rows = iter_hf(repo, split=split, config=src.get("config"))
        w_ = Writer(MIX / src["cat"])
        for row in rows:
            w_.add(extract_text(row) if isinstance(row, dict) else str(row))
            if w_.size >= limit_bytes:
                break
        w_.flush()
    except Exception as e:
        tag = " (fallback)" if src.get("fallback") else ""
        print(f"  [skip] {src['cat']} {repo}{tag}: {type(e).__name__}: {str(e)[:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-gb", type=float, default=120.0)
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--source-timeout", type=int, default=7200)
    ap.add_argument("--clean", action="store_true",
                    help="wipe previous mix/ before building (default: reuse as cache)")
    ap.add_argument("--include-distillation", action="store_true", default=True)
    ap.add_argument("--no-distillation", dest="include_distillation", action="store_false")
    ap.add_argument("--peek", action="store_true")
    args = ap.parse_args()

    if args.mini:
        args.total_gb = 0.05
        args.source_timeout = 300

    if args.peek:
        for s in SOURCES:
            if s["kind"] in ("hf", "code"):
                try:
                    ds = load_dataset(s["repo"], split=s.get("split", "train"), streaming=True)
                    print(f"{s['repo']} [{s.get('split','train')}]: keys={list(next(iter(ds)).keys())}")
                except Exception as e:
                    print(f"{s['repo']}: {type(e).__name__}: {str(e)[:70]}")
        return

    if args.include_distillation and any(s.get("distill") for s in SOURCES):
        print("WARNING: distillation sets contain outputs of frontier models "
              "(Claude/GPT/Gemini/Grok/Qwen). Their ToS may forbid training on outputs. "
              "Included per user decision (non-commercial use).")

    # ponytail: mix/ is a persistent local cache by default — re-runs top up
    # what's missing instead of re-streaming. --clean wipes for a fresh build.
    remaining = {c: max(0.0, CAT_BUDGET[c] * args.total_gb * GB - cat_bytes(c)) for c in CAT_BUDGET}
    if args.clean:
        for d in MIX.rglob("*.txt"):
            d.unlink()
        remaining = {c: CAT_BUDGET[c] * args.total_gb * GB for c in CAT_BUDGET}
        print("[clean] wiped previous mix/")
    else:
        print("[cache] reusing existing mix/ where present (top-up mode)")
    print(f"Building ~{args.total_gb:.1f} GB into {MIX}")
    for src in SOURCES:
        cat = src["cat"]
        if remaining[cat] <= 0:
            continue
        if src.get("distill") and not args.include_distillation:
            continue
        label = "local arxiv-ai" if src["kind"] == "arxiv" else (
            f"{src['repo']} (per-language)" if src["kind"] == "code" else src["repo"])
        print(f"  -> {cat:9} {label}  limit {remaining[cat]/GB:.3f} GB", flush=True)
        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(args.source_timeout)
            source_rows(src, limit_bytes=remaining[cat])
        except SourceTimeout:
            print(f"  [timeout] {cat} {label}: exceeded {args.source_timeout}s, skipping", flush=True)
        finally:
            signal.alarm(0)
        remaining[cat] = max(0.0, CAT_BUDGET[cat] * args.total_gb * GB - cat_bytes(cat))

    man = {}
    for d in MIX.rglob("*.txt"):
        cat = d.parent.name if d.parent.parent != MIX / "code" else "code"
        man.setdefault(cat, {"files": 0, "bytes": 0})
        man[cat]["files"] += 1
        man[cat]["bytes"] += d.stat().st_size
    print("\n=== manifest ===", flush=True)
    tot = 0
    for cat in sorted(man):
        b = man[cat]["bytes"]
        tot += b
        print(f"  {cat:9} {man[cat]['files']:5} files  {b/GB:.3f} GB", flush=True)
    print(f"  TOTAL     {tot/GB:.3f} GB", flush=True)
    (MIX / "manifest.json").write_text(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
