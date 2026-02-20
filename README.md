# Instacart Next-Order Recommendation

Proof-of-concept for **next-order product recommendation** using the [Instacart orders dataset](https://www.kaggle.com/c/instacart-market-basket-analysis): a two-tower Sentence-BERT model where the **anchor** is user context (past orders + optional order pattern) and the **positive** is product text (name, aisle, department). Training uses `MultipleNegativesRankingLoss`; at serve time we encode the user context and rank products by cosine similarity.

---

## What we are predicting

- **Target:** For each user, we predict **which products are most likely to appear in their next order**. We do *not* predict the exact next order (e.g. a single basket); we produce a **ranking over the full product catalog** so that items the user is likely to buy next appear at the top.
- **Formally:** Given a user’s **prior order history** (products bought, order timing, gaps between orders), the model outputs a **score for every product** (or a **top-k list**). Products with higher scores are more likely to be in the user’s next basket. Evaluation uses standard retrieval metrics (Accuracy@k, MRR, NDCG, MAP) against the actual next-order products as relevance labels.
- **No leakage:** The model never sees the next order at prediction time. Training uses (anchor = prior-only context, positive = one product from the next order); at serve and in evaluation, the query is the same prior-only context, so we simulate a realistic “what should we recommend *now*?” setting.

---

## How this model could be used later

- **In-app “reorder” or “buy again”:** Surface a short list of products (“You might need these”) on the home screen or before checkout, ordered by the model’s scores. Users can one-tap add items they regularly buy.
- **Email / push:** Trigger “Your usual items are back” or “Restock these?” campaigns using top-k recommendations per user, optionally filtered by category or recency.
- **Cold start:** For users with few or no prior orders, the context can be minimal (e.g. first order only); the same pipeline still returns a ranking. You can combine with non-personalized fallbacks (trending, category) when context is too weak.
- **APIs and services:** Expose the recommender as a service: input = user context string (or user_id + lookup from your DB), output = top-k product IDs and scores. Other teams (search, ads, merchandising) can call it to personalize surfaces.
- **Batch precompute:** Precompute top-k per user on a schedule (e.g. nightly), store in a key-value store or feature store, and serve from cache for low-latency UX while retraining the model periodically.
- **A/B testing:** Run the two-tower model as one arm (e.g. “SBERT next-order”) vs rule-based or other models, and measure impact on add-to-cart, order size, or repeat purchase.

---

## Requirements

- Python 3.10+
- Instacart dataset (e.g. from Kaggle)

---

## Setup

```bash
uv sync
```

Or `pip install -e .`. Optional: add a `.env` in the project root with `HF_TOKEN` if you use private Hugging Face assets.

---

## Data

Place the following under `data/` (or pass `--data-dir` when running data prep):

- **orders.csv** — order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order (used to split train vs prior and build user context).
- **order_products__prior.csv** — order_id, product_id (prior orders only; used to build order→products and anchor/positive pairs).
- **products.csv** — product_id, product_name, aisle_id, department_id.
- **aisles.csv**, **departments.csv** — names for product text.

Data prep writes to **processed/** in a param-based subdir (e.g. `processed/p5_mp20_ef0.1/`) so different settings don’t overwrite each other. That directory contains `train_dataset/`, `eval_dataset/`, `eval_queries.json`, `eval_corpus.json`, `eval_relevant_docs.json`, and `data_prep_params.json`.

---

## Pipeline

1. **Prepare** — build (anchor, positive) pairs and eval IR artifacts.
2. **Train** — two-tower SBERT with `MultipleNegativesRankingLoss` and optional `InformationRetrievalEvaluator`.
3. **Serve** — load model + corpus, encode context, return top-k by cosine similarity.

### Commands

```bash
# Prepare (writes to processed/<param_subdir>/)
uv run python -m src.data.prepare_instacart_sbert

# Train (default: auto-resolve processed dir to single param subdir if needed)
uv run python -m src.train.train_sbert --lr 1e-4
# Or: --processed-dir processed/p5_mp20_ef0.1

# Serve (CLI and Python API: load_recommender, recommend)
uv run python -m src.inference.serve_recommendations --top-k 10
# Or: --corpus processed/p5_mp20_ef0.1/eval_corpus.json --model-dir models/two_tower_sbert/final --query "..."
```

---

## Prediction problem (summary)

- **Task:** Rank the catalog so products in the user’s *next* order are at the top (see **What we are predicting** above).
- **Input:** User context from *prior* orders only (products, day/hour, gaps; no next-order info).
- **Output:** Ranking over the catalog (top-k by cosine similarity).
- **Train vs serve:** (anchor, positive) = (prior-only context, product in next order). At serve time the “Next: …” part is dropped so evaluation matches production (we don’t know the next order time at request time).

---

## Results

**Data (example from data prep, e.g. max_prior_orders=5, max_product_names=20, eval_frac=0.1):**

| Train pairs | Eval pairs | Eval queries | Corpus size |
|-------------|------------|--------------|-------------|
| ~1.25M      | ~138k      | ~13k         | ~50k        |

**Example evaluation metrics** (default setup: `processed/p5_mp20_ef0.1`, `all-MiniLM-L6-v2`, `max_seq_length` 256, eval on ~13k queries over ~50k corpus). Metrics from the built-in `InformationRetrievalEvaluator`:

| Metric       | After 1 epoch | After 3 epochs |
|--------------|----------------|----------------|
| Accuracy@1   | 0.106–0.11     | ~0.19          |
| Accuracy@10  | 0.25–0.29      | ~0.29–0.46     |
| MRR@10       | 0.15–0.16      | ~0.27          |
| NDCG@10      | 0.066–0.078    | ~0.12          |
| MAP@100      | 0.039–0.047    | —              |

So after one epoch the model puts the correct product in the top-10 for about **25–29%** of eval queries; after a few more epochs that can reach the **~30–46%** range depending on batch size and learning rate. Best checkpoint is saved by **NDCG@10** when the IR evaluator is enabled.

**IR metrics:** Accuracy@1, Accuracy@10, MRR@10, NDCG@10, MAP@100 (from `InformationRetrievalEvaluator`). Enable by default; disable with `--no-information-retrieval-evaluator` for faster training.

**Sample usage:** Run the serve script without `--query` for the built-in demo, or use `--eval-query-id <order_id>` to run on a query from `eval_queries.json`.

---

### Demo inference

Demo query (past orders only; no “Next order” at serve time):

```
[+7d w4h14] Organic Milk, Whole Wheat Bread.
```

Example top-5 output:

```
Top-5 recommendations:
  1. product_id=48628 (score=0.7877) Product: Organic Whole Wheat Bread. Aisle: bread. Department...
  2. product_id=13517 (score=0.7850) Product: Whole Wheat Bread. Aisle: bread. Department: bakery...
  3. product_id=44103 (score=0.7241) Product: Honey Whole Wheat Bread. Aisle: bread. Department: ...
  4. product_id=6454 (score=0.7207) Product: Whole Wheat Bread Loaf. Aisle: bread. Department: b...
  5. product_id=38591 (score=0.6624) Product: Organic Whole Wheat. Aisle: bread. Department: bake...
```

Scores are cosine similarity in [0, 1]. Use `--eval-query-id` to test on a real eval order.

---

## Training notes

- Training is slow (e.g. multiple hours for 5 epochs). Larger base models or longer `--max-seq-length` increase runtime and memory. Current defaults (`all-MiniLM-L6-v2`, `max_seq_length` 256) are set for feasibility on typical hardware.
- **Why ~1s+ per step (e.g. on Apple Silicon):** On MPS the code uses `dataloader_num_workers=0` (and no fp16) for stability. Data loading and tokenization run on the main thread, so the GPU often waits for the next batch and there’s no prefetch. On CUDA you can use `--dataloader-num-workers` for faster steps. With **dynamic padding** (pad to longest in batch), most steps can be faster, but when the batch length (and thus tensor shape) changes, MPS may recompile and you can see steps over 5s; length bucketing would limit recompilation to a few shapes.
- No gradient accumulation; one batch per step.

---

## Project structure

```
data/                  # Raw CSVs (not in repo)
processed/<param>/      # Data prep output (train_dataset, eval_*, data_prep_params.json)
models/two_tower_sbert/  # Checkpoints and final/
src/
  constants.py         # PROJECT_ROOT, default paths
  utils.py             # setup_colored_logging(), resolve_processed_dir()
  data/
    prepare_instacart_sbert.py
  train/
    train_sbert.py
  inference/
    serve_recommendations.py
notebooks/             # prepare_instacart_sbert, train_sbert, serve_recommendations
pyproject.toml, uv.lock
```

---

## License

MIT. Use of the Instacart dataset is subject to its own terms (e.g. Kaggle).
