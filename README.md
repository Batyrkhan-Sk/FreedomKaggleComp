# KaggleComp

My solutions to a private three-task Kaggle competition. Each task is a different kind of ML problem.

| Task | Problem | Approach | Main notebook |
|---|---|---|---|
| **Lost in the Museum** | Match blurry, tilted, cropped visitor photos of paintings to their clean gallery scans | Embed every image with DINOv3 (ViT-7B), then whiten the embeddings, with no training | `kaggle-museum-solution-7b.ipynb` |
| **Personalized Music Recommender** | Recommend the top 50 tracks for each user | Build a candidate pool per user (listening history + co-visitation), then rank it with gradient boosting | `music/music-recommender-solution.ipynb` |
| **Who's the Best Pitcher** | Answer 220 questions about MLB data stored in large JSON files | Resolve each question to a leaf in the JSON and copy its value exactly, since answers are scored by exact string match | `pitcher/pitcher-solution-v7.ipynb` |

## Layout

- **Root** holds the museum task: feature extraction, fusion and validation scripts, plus the Kaggle notebooks.
- `finetune/` has fine-tuning experiments for the museum task.
- `music/` holds the recommender, including candidate generation, ranking models and the experiments behind them.
- `pitcher/` holds the question-answering pipeline and LLM re-ranking experiments.

The `*-solution*` notebooks are the final, reproducible versions. Most other scripts are experiments.

## Running

The notebooks are meant to run on Kaggle with the competition data attached. The local scripts use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python <script>.py
```

`music/` has its own `pyproject.toml`; run `uv sync` inside it too.

**Competition data is not included.** The competition is private and its data is confidential, so none of it is in this repo.
