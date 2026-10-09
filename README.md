# GTArena evaluation code

This code runs a model on the three GTArena question pools through an OpenAI-compatible chat-completions API and scores its answers as the paper does. It uses only the Python standard library and runs on Python 3.9 or newer.

## Get the data

```
huggingface-cli download konkazzz/GTArena --repo-type dataset --local-dir gtarena
```

## Run a model

```
export API_KEY=...
python run.py test_intention  --data gtarena --model MODEL --base-url BASE_URL --api-key-env API_KEY --out ti.jsonl
python run.py task_execution  --data gtarena --model MODEL --base-url BASE_URL --api-key-env API_KEY --out te.jsonl
python run.py defect_judgment --data gtarena --model MODEL --base-url BASE_URL --api-key-env API_KEY --out dj.jsonl
```

`--base-url` is the API root that `/chat/completions` is appended to. `--api-key-env` names the environment variable that holds the key, and a local server that needs no key can go without it. Each item is one request whose message is the prompt followed by the item's images. The request sets temperature 0, and an endpoint that rejects the parameter is asked again at its default. `--no-screenshot` runs test intention without the screenshot, and that run needs its own output file. Running a command again resumes it. `--max-completion-tokens` (default 8000) sets the answer budget and `--workers` (default 4) the number of parallel requests.

A reply is read as JSON. An empty reply, or one that does not validate, is asked again, up to three attempts. In task execution and defect judgment, a last reply without the JSON object still counts when it names exactly one action or one label.

## Judge test intention

```
python judge.py --data gtarena --answers ti.jsonl --out ti_judged.jsonl --base-url BASE_URL --api-key-env API_KEY
```

Three judges, `gpt-5.5`, `gemini-3.1-pro` and `deepseek-v4-pro`, read each answered item's defect description and the model's five checks, without the screenshot. For each check, a judge says whether running it would expose the defect. To serve each judge from its own endpoint, give `--judge MODEL BASE_URL KEY_VARIABLE` three times.

## Score

```
python score.py test_intention  --data gtarena --answers ti.jsonl --judgments ti_judged.jsonl
python score.py task_execution  --data gtarena --answers te.jsonl
python score.py defect_judgment --data gtarena --answers dj.jsonl
```

- **Coverage** (test intention): the share of the 114 items for which two of the three judges say that the same check would expose the item's defect.
- **Exact match** (task execution): the share of the 788 questions marked `exact_match_scored` whose answer has the action type and the target of one of the accepted answers. The target is the control, the typed text, the key, the drag direction or the app to open.
- **Accuracy, recall and specificity** (defect judgment): over all 1,858 items, the share judged correctly, the share of defective items called defect, and the share of clean items called clean.

An unanswered or malformed answer counts as wrong.

## License

MIT
