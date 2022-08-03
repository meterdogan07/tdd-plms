# Pre-process Default Documents (Corpus)
**pre-process.py**  script converts to each `tdd-corpus-XX.txt` file to loose json format as:
```
{"src": "The Internet", "text": "The quick brown fox", "type": "Tur", "id": "0", "title": "tdd-corpus-XX"}
```
You need run a seperate script for each `tdd-corpus-XX.txt`.

## Requirments
```
pip install tqdm
```

## How to Run
```
python pre-process.py --file_path inpath/tdd-corpus-XX.txt --outdir outpath/
```
