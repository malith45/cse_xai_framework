"""M3 Trilingual Sentiment Engine - Kaggle GPU job (§4.6.4, Table 4.8). Needs internet ON.

Steps
 1. Load the scraped articles (news_scraper.py output) and route each by language: Unicode script
    for Sinhala/Tamil; fastText lid.176 for Latin-script text (English vs code-mixed).
 2. Fine-tune XLM-RoBERTa base (3 classes) by transfer from: Senevirathne et al. (2020) Sinhala news
    comments, DravidianCodeMix Tamil-English (Chakravarthi et al., 2022) and Financial PhraseBank
    (Malo et al., 2014). 2 epochs, lr 2e-5, max length 128, 10% held out for a sanity check.
 3. Score: English -> FinBERT (ProsusAI/finbert). Sinhala, Tamil, code-mixed -> fine-tuned XLM-R.
    If XLM-R's top-class probability < 0.60 for a Sinhala/Tamil article, translate it to English with
    NLLB-200 distilled 600M and re-score with FinBERT (secondary score). s = P(pos) - P(neg).
    The score used downstream ("score") is the FinBERT-on-translation score when the fallback fired,
    otherwise the primary score; both are stored.
 4. Link companies (alias dictionary, D6) and flag market-wide news.
Outputs (/kaggle/working): articles_scored.parquet, xlmr_holdout_report.json, xlmr_model/ (optional)

Run in a Kaggle notebook cell:
    !python /kaggle/input/<code-dataset>/code/scripts/kaggle_sentiment.py --news /kaggle/input/<news-dataset> --out /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment.text import is_market_news, link_companies, script_language  # noqa: E402

SEED = 42
FALLBACK_THRESHOLD = 0.60
LABELS = ["negative", "neutral", "positive"]


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=False)


# --------------------------------------------------------------------------- data

def load_articles(news: Path) -> pd.DataFrame:
    frames = []
    # article files only: adaderana_biz_en.csv and hiru_<lang>_articles.csv (not the *_refetched.csv
    # side files, whose bodies are already merged into the article files, nor the headline lists)
    for f in sorted(news.rglob("*.csv")):
        if not (f.name.startswith("adaderana") or f.name.endswith("_articles.csv")):
            continue
        d = pd.read_csv(f, dtype=str)
        frames.append(d)
    arts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # relevant headlines whose article page could not be fetched are scored on the headline alone
    heads = []
    for f in sorted(news.rglob("*headlines.csv")):
        h = pd.read_csv(f, dtype=str)
        heads.append(h[h["relevant"] == "1"])
    if heads:
        h = pd.concat(heads, ignore_index=True)
        h = h[~h["url"].isin(arts["url"] if len(arts) else [])]
        h["body"] = ""
        arts = pd.concat([arts, h], ignore_index=True)
    arts = arts.drop_duplicates("url").reset_index(drop=True)
    arts["title"] = arts["title"].fillna("")
    arts["body"] = arts["body"].fillna("")
    # model input: headline + lead (first ~600 characters of the body)
    arts["text"] = (arts["title"] + ". " + arts["body"].str[:600]).str.strip(". ")
    arts["published"] = arts.get("page_datetime", pd.Series(dtype=str)).where(
        arts.get("page_datetime", pd.Series(dtype=str)).fillna("").str.contains(r"\d{1,2}:\d{2}"), arts["published_at"])
    return arts


class _NumpyCompat:
    """fasttext 0.9.x calls np.array(..., copy=False), which NumPy 2 rejects. Route that call to
    np.asarray (same behaviour as NumPy 1) and pass everything else through."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def array(obj, *args, copy=None, **kw):
        return np.asarray(obj) if copy is False else np.array(obj, *args, **kw)


def route_language(arts: pd.DataFrame) -> pd.Series:
    try:
        import fasttext
    except ImportError:
        sh("pip -q install fasttext-wheel")
        import fasttext
    import fasttext.FastText as _ft
    _ft.np = _NumpyCompat()
    model_path = Path("/kaggle/working/lid.176.ftz")
    if not model_path.exists():
        sh(f"wget -q -O {model_path} https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz")
    lid = fasttext.load_model(str(model_path))
    out = []
    for t in arts["text"]:
        s = script_language(t)
        if s in ("si", "ta"):
            out.append(s)
        elif s == "latin":
            labels, probs = lid.predict(t.replace("\n", " ")[:1000], k=1)
            out.append("en" if labels[0] == "__label__en" and probs[0] >= 0.60 else "code_mixed")
        else:
            out.append("unknown")
    return pd.Series(out, index=arts.index)


def transfer_data() -> pd.DataFrame:
    rows = []
    # Senevirathne et al. (2020): Lankadeepa + GossipLanka comments (Google Drive links in the paper repo)
    sh("pip -q install gdown")
    import gdown
    for fid, name in [("1e9xRaKoK3wgulCj7uF1mSblV2UB4D95B", "lankadeepa.csv"), ("1Y24oCg0oWINSo5Om7cJWBy9f03cMjGzX", "gossiplanka.csv")]:
        p = Path("/kaggle/working") / name
        # a copy uploaded as a Kaggle dataset (Google Drive often refuses automated downloads) wins
        local = [f for f in Path("/kaggle/input").rglob("*.csv") if f.stem.lower().startswith(name.split(".")[0])]
        if local:
            p = local[0]
        try:
            if not p.exists():
                gdown.download(id=fid, output=str(p), quiet=True)
            d = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
            text_col = max((c for c in d.columns if d[c].dtype == object), key=lambda c: d[c].astype(str).str.len().mean())
            lab_col = next(c for c in d.columns if set(d[c].astype(str).str.upper().unique()) & {"POSITIVE", "NEGATIVE", "NEUTRAL"})
            m = {"POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral"}
            lab = d[lab_col].astype(str).str.upper().map(m)
            rows.append(pd.DataFrame({"text": d[text_col].astype(str), "label": lab, "source": "senevirathne_si"}).dropna())
        except Exception as e:  # noqa: BLE001
            print("Sinhala corpus failed:", name, e, flush=True)
    # DravidianCodeMix Tamil-English
    try:
        if not Path("/kaggle/working/dcm").exists():
            sh("git clone -q --depth 1 https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset /kaggle/working/dcm")
        import csv
        # the file is TAB-separated (label<TAB>text); read as comma-separated it yields no usable rows
        d = pd.read_csv("/kaggle/working/dcm/programs/tamil_sentiment_full.csv", sep="\t", header=None,
                        names=["label", "text"], on_bad_lines="skip", quoting=csv.QUOTE_NONE, dtype=str)
        m = {"Positive": "positive", "Negative": "negative", "unknown_state": "neutral"}   # Mixed_feelings, not-Tamil dropped
        d["label"] = d["label"].str.strip().map(m)
        rows.append(d.dropna().assign(source="dravidiancodemix_ta"))
    except Exception as e:  # noqa: BLE001
        print("Tamil corpus failed:", e, flush=True)
    # Financial PhraseBank (Malo et al., 2014), 50%-agreement sentences. Newer `datasets` releases no longer
    # run loading scripts, so fall back to the original zip / the parquet conversion on the Hub.
    fp = None
    try:
        from datasets import load_dataset
        fp = load_dataset("takala/financial_phrasebank", "sentences_50agree", trust_remote_code=True)["train"].to_pandas()
        fp["label"] = fp["label"].map({0: "negative", 1: "neutral", 2: "positive"})
        fp = fp.rename(columns={"sentence": "text"})
    except Exception as e:  # noqa: BLE001
        print("PhraseBank via datasets failed:", e, flush=True)
    if fp is None:
        try:
            import io
            import urllib.request
            import zipfile
            url = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
            name = next(n for n in z.namelist() if n.endswith("Sentences_50Agree.txt"))
            lines = z.read(name).decode("latin-1").splitlines()
            pairs = [ln.rsplit("@", 1) for ln in lines if "@" in ln]
            fp = pd.DataFrame(pairs, columns=["text", "label"])
            fp["label"] = fp["label"].str.strip()
        except Exception as e:  # noqa: BLE001
            print("PhraseBank zip failed:", e, flush=True)
    if fp is None:
        try:
            fp = pd.read_parquet("https://huggingface.co/datasets/takala/financial_phrasebank/resolve/"
                                 "refs%2Fconvert%2Fparquet/sentences_50agree/train/0000.parquet")
            fp["label"] = fp["label"].map({0: "negative", 1: "neutral", 2: "positive"})
            fp = fp.rename(columns={"sentence": "text"})
        except Exception as e:  # noqa: BLE001
            print("PhraseBank parquet failed:", e, flush=True)
    if fp is not None:
        rows.append(fp[["text", "label"]].dropna().assign(source="phrasebank_en"))
    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["text", "label", "source"])
    data = data[data["text"].astype(str).str.strip().str.len() > 3]
    have = set(data["source"].unique())
    need = {"senevirathne_si", "dravidiancodemix_ta", "phrasebank_en"}
    if need - have:   # stop before any GPU time is spent rather than train a model without one language
        raise SystemExit(f"transfer corpus missing: {sorted(need - have)} - send this message to Claude")
    # cap each source at 12,000 rows so no single corpus dominates
    data = data.groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), 12000), random_state=SEED))
    print(data.groupby(["source", "label"]).size(), flush=True)
    return data.sample(frac=1, random_state=SEED).reset_index(drop=True)


# --------------------------------------------------------------------------- models

def finetune_xlmr(data: pd.DataFrame, out: Path):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3).to(dev)
    y = data["label"].map({l: i for i, l in enumerate(LABELS)}).values
    n_hold = int(0.1 * len(data))
    tr_idx, ho_idx = np.arange(n_hold, len(data)), np.arange(n_hold)

    def batches(idx, bs, shuffle):
        if shuffle:
            idx = np.random.default_rng(SEED).permutation(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            enc = tok(list(data["text"].values[b]), truncation=True, max_length=128, padding=True, return_tensors="pt")
            yield {k: v.to(dev) for k, v in enc.items()}, torch.tensor(y[b], device=dev)

    epochs, bs = 2, 32
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = epochs * (len(tr_idx) // bs + 1)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler()
    for ep in range(epochs):
        model.train()
        for i, (enc, yy) in enumerate(batches(tr_idx, bs, True)):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**enc, labels=yy).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            if i % 200 == 0:
                print(f"epoch {ep} step {i} loss {loss.item():.3f}", flush=True)
    # hold-out sanity check
    model.eval()
    preds = []
    with torch.no_grad():
        for enc, _ in batches(ho_idx, 64, False):
            preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    from sklearn.metrics import classification_report
    ho = data.iloc[ho_idx].assign(pred=[LABELS[p] for p in preds])
    rep = {src: classification_report(g["label"], g["pred"], output_dict=True, zero_division=0)
           for src, g in ho.groupby("source")}
    json.dump(rep, open(out / "xlmr_holdout_report.json", "w"), indent=1)
    print({k: round(v["macro avg"]["f1-score"], 3) for k, v in rep.items()}, flush=True)
    return tok, model


def classify(texts, tok, model, bs=64, max_len=256):
    import torch
    dev = next(model.parameters()).device
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(list(texts[i:i + bs]), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", required=True)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--save-model", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sh("pip -q install sentencepiece")   # fasttext is preinstalled on Kaggle; installed on demand otherwise
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

    transfer = transfer_data()            # first: fail fast (before GPU work) if a training corpus is missing
    arts = load_articles(Path(a.news))
    arts["lang"] = route_language(arts)
    print(arts["lang"].value_counts(), flush=True)

    # English: FinBERT
    fb_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fb = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to("cuda")
    id2 = {i: l.lower() for i, l in fb.config.id2label.items()}

    def finbert_probs(texts):
        p = classify(texts, fb_tok, fb, max_len=256)
        return pd.DataFrame({id2[i]: p[:, i] for i in range(p.shape[1])})[LABELS].values

    arts[["p_neg", "p_neu", "p_pos"]] = np.nan
    arts["model"] = ""
    en = arts["lang"] == "en"
    if en.any():
        arts.loc[en, ["p_neg", "p_neu", "p_pos"]] = finbert_probs(arts.loc[en, "text"].values)
        arts.loc[en, "model"] = "finbert"

    # Sinhala / Tamil / code-mixed: fine-tuned XLM-R
    tok, xlmr = finetune_xlmr(transfer, out)
    if a.save_model:
        xlmr.save_pretrained(out / "xlmr_model")
        tok.save_pretrained(out / "xlmr_model")
    ml = arts["lang"].isin(["si", "ta", "code_mixed"])
    if ml.any():
        arts.loc[ml, ["p_neg", "p_neu", "p_pos"]] = classify(arts.loc[ml, "text"].values, tok, xlmr, max_len=256)
        arts.loc[ml, "model"] = "xlm-roberta"
    arts["score_primary"] = arts["p_pos"] - arts["p_neg"]

    # NLLB fallback for low-confidence Sinhala/Tamil articles
    low = arts["lang"].isin(["si", "ta"]) & (arts[["p_neg", "p_neu", "p_pos"]].max(axis=1) < FALLBACK_THRESHOLD)
    arts["translation_en"], arts["score_secondary"] = "", np.nan
    print("fallback articles:", int(low.sum()), flush=True)
    if low.any():
        nt = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        nm = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=torch.float16).to("cuda")
        trans = []
        for lang, code in (("si", "sin_Sinh"), ("ta", "tam_Taml")):
            idx = arts.index[low & (arts["lang"] == lang)]
            nt.src_lang = code
            for i in range(0, len(idx), 16):
                b = idx[i:i + 16]
                enc = nt(list(arts.loc[b, "text"].str[:500]), return_tensors="pt", truncation=True, max_length=256, padding=True).to("cuda")
                with torch.no_grad():
                    gen = nm.generate(**enc, forced_bos_token_id=nt.convert_tokens_to_ids("eng_Latn"), max_new_tokens=200)
                trans += list(zip(b, nt.batch_decode(gen, skip_special_tokens=True)))
        for i, t in trans:
            arts.at[i, "translation_en"] = t
        p = finbert_probs(arts.loc[low, "translation_en"].values)
        arts.loc[low, "score_secondary"] = p[:, 2] - p[:, 0]
    arts["score"] = arts["score_secondary"].where(low, arts["score_primary"])
    arts["fallback_used"] = low

    arts["companies"] = arts["text"].map(lambda t: "|".join(link_companies(t)))
    arts["market_news"] = arts["text"].map(is_market_news)
    keep = ["source", "language_hint", "lang", "url", "published_at", "published", "title", "model", "p_neg", "p_neu",
            "p_pos", "score_primary", "score_secondary", "score", "fallback_used", "translation_en", "companies", "market_news"]
    arts[keep].to_parquet(out / "articles_scored.parquet")
    print("saved", len(arts), flush=True)


if __name__ == "__main__":
    main()
