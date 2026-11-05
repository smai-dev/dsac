# Linux setup

Run from the extracted `web_content_insights/` project folder. The fixed CSV is included, so the scraper is not necessary for the analysis notebooks.

## Python environment

On Debian or Ubuntu:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then select `Python (Web Content Insights)` as the VS Code notebook kernel.

## NLTK resources

The analysis notebooks already call `nltk.download` for resources. If downloads are blocked, install the resources using the active environment:

```bash
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords wordnet
```

## Memory and model-search notes

The source SVD sweep and hierarchical models can consume significant memory. The source grid-search silhouette scorer is not valid as supplied for an unsupervised estimator, so the corresponding cell may need a separate modeling fix before a full end-to-end run. This package preserves that computational choice and does not claim those cells run successfully.
