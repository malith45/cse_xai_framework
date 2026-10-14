"""Create the two Kaggle notebooks (GPU + Internet ON) the author runs: LSTM/GRU walk-forward and the
trilingual sentiment engine. Output: dist/kaggle/*.ipynb"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "kaggle"
OUT.mkdir(parents=True, exist_ok=True)


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    return n


rnn = nb([
    ("md", "# CSE dissertation - LSTM/GRU walk-forward (M4) + expected-gradient SHAP (M5)\n"
           "Settings (right panel): **Accelerator = GPU P100**, **Internet = On**. Add the private dataset "
           "`cse-rnn-input` (the `rnn_input` folder from Claude). Then click **Save Version -> Save & Run All (Commit)**; "
           "it runs in the background (about 2-5 hours) and you can close the browser.\n\n"
           "When it finishes, download `/kaggle/working/rnn_output.zip` (Output tab) and put it in "
           "`Research Code Files\\Kaggle Outputs\\`."),
    ("code", "!pip -q install optuna shap 2>/dev/null; nvidia-smi -L"),
    ("code", "import glob, os, shutil\nrun = glob.glob('/kaggle/input/**/code/scripts/run_rnn.py', recursive=True)[0]\n"
             "src = os.path.dirname(os.path.dirname(run))\n"
             "shutil.copytree(src, '/kaggle/working/code', dirs_exist_ok=True)\n"
             "data = os.path.dirname(glob.glob('/kaggle/input/**/panel_universe.parquet', recursive=True)[0])\n"
             "print('code:', src, '| data:', data)"),
    ("code", "!cd /kaggle/working/code && python scripts/run_rnn.py --data {data} --out /kaggle/working/rnn_out"),
    ("code", "!cd /kaggle/working && rm -f rnn_output.zip && zip -qr rnn_output.zip rnn_out && ls -la rnn_output.zip"),
])
nbf.write(rnn, OUT / "cse_rnn_walkforward.ipynb")

sent = nb([
    ("md", "# CSE dissertation - trilingual sentiment engine (M3)\n"
           "Settings: **Accelerator = GPU P100**, **Internet = On**. Add two private datasets: `cse-news` "
           "(your `news_raw` folder from the scraper) and `cse-sentiment-code` (folder `dist/kaggle/sentiment_code`). "
           "Run All (about 1-2 hours). Download `/kaggle/working/sentiment_output.zip` and put it in "
           "`Research Code Files\\Kaggle Outputs\\`."),
    ("code", "import glob, os, shutil\nrun = glob.glob('/kaggle/input/**/code/scripts/kaggle_sentiment.py', recursive=True)[0]\n"
             "shutil.copytree(os.path.dirname(os.path.dirname(run)), '/kaggle/working/code', dirs_exist_ok=True)\n"
             "news = os.path.dirname(glob.glob('/kaggle/input/**/log.txt', recursive=True)[0])\n"
             "print('news folder:', news)"),
    ("code", "!cd /kaggle/working/code && python scripts/kaggle_sentiment.py --news {news} --out /kaggle/working/sent_out"),
    ("code", "!cd /kaggle/working && rm -f sentiment_output.zip && zip -qr sentiment_output.zip sent_out/*.parquet sent_out/*.json && ls -la sentiment_output.zip"),
])
nbf.write(sent, OUT / "cse_sentiment.ipynb")
print("written", list(OUT.glob("*.ipynb")))
