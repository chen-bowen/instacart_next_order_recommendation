# Instacart Next-Order Recommendation

Proof-of-concept for **next-order product recommendation** using the [Instacart orders dataset](https://www.kaggle.com/c/instacart-market-basket-analysis): a two-tower Sentence-BERT model where the **anchor** is user context (past orders + optional order pattern) and the **positive** is product text (name, aisle, department). Training uses `MultipleNegativesRankingLoss`; at serve time we encode the user context and rank products by cosine similarity.

---

## What we are predicting

- **Target:** For each user, we predict **which products are most likely to appear in their next order**. We do _not_ predict the exact next order (e.g. a single basket); we produce a **ranking over the full product catalog** so that items the user is likely to buy next appear at the top.
- **Formally:** Given a user’s **prior order history** (products bought, order timing, gaps between orders), the model outputs a **score for every product** (or a **top-k list**). Products with higher scores are more likely to be in the user’s next basket. Evaluation uses standard retrieval metrics (Accuracy@k, MRR, NDCG, MAP) against the actual next-order products as relevance labels.
- **No leakage:** The model never sees the next order at prediction time. Training uses (anchor = prior-only context, positive = one product from the next order); at serve and in evaluation, the query is the same prior-only context, so we simulate a realistic “what should we recommend _now_?” setting.

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

- **Python** 3.10+ (3.12 recommended; managed via `uv` or your environment).
- **Instacart dataset** from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data) (orders, order_products, products, aisles, departments). Download and place CSVs under `data/`.
- **Disk:** ~2–3 GB for raw data + processed datasets; model checkpoints add a few hundred MB per run.
- **Memory:** 8 GB RAM is enough for data prep and inference; training benefits from 16 GB+ and a GPU (CUDA or Apple MPS) for speed.

---

## Setup

1. **Clone or open the repo** and enter the project root.

2. **Install dependencies** (prefer `uv` for a locked environment):

   ```bash
   uv sync
   ```

   Or with pip: `pip install -e .` (see `pyproject.toml` for dependencies).

3. **Download the Instacart data** from Kaggle into `data/`. You need at least:
   - `orders.csv`
   - `order_products__prior.csv`
   - `products.csv`
   - `aisles.csv`
   - `departments.csv`

4. **Optional:** Create a `.env` file in the project root with `HF_TOKEN=...` if you use private Hugging Face models or datasets.

5. **Verify:** Run data prep (see Pipeline below); it will fail with a clear error if any CSV is missing or misnamed.

---

## Data

### Input files (under `data/`)

| File                            | Key columns                                                                                         | Role                                                                                                                          |
| ------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **orders.csv**                  | order_id, user_id, **eval_set**, order_number, order_dow, order_hour_of_day, days_since_prior_order | `eval_set == "train"` → target “next” orders we predict for; `eval_set == "prior"` → history used to build user context only. |
| **order_products\_\_prior.csv** | order_id, product_id                                                                                | Which products are in each prior order; used to build (anchor, positive) pairs.                                               |
| **products.csv**                | product_id, product_name, aisle_id, department_id                                                   | Product names and hierarchy.                                                                                                  |
| **aisles.csv**                  | aisle_id, aisle                                                                                     | Aisle names for product text.                                                                                                 |
| **departments.csv**             | department_id, department                                                                           | Department names for product text.                                                                                            |

No `order_products__train.csv` is required for this pipeline: we only use prior orders for context and the train-set orders to define _which_ next order we are predicting (and to split train/eval by order).

### Data prep output (processed/)

Data prep writes under a **param-based subdir** of `processed/`, e.g. `processed/p5_mp20_ef0.1/`, so different runs (e.g. different `max_prior_orders` or `eval_frac`) do not overwrite each other. The subdir name encodes: `p` = max_prior_orders, `mp` = max_product_names, `ef` = eval_frac (and optionally `sf` = sample_frac, `no_serve` if eval queries keep “Next: …”).

| Output                      | Description                                                                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **train_dataset/**          | Hugging Face Dataset on disk: columns `anchor`, `positive`. Each row is one (user context, product from next order) pair for training. |
| **eval_dataset/**           | Same format; used for validation loss (optional).                                                                                      |
| **eval_queries.json**       | Map from query_id (order_id as string) to the **serve-time** user context string (no “Next: …” when `eval_serve_time=True`).           |
| **eval_corpus.json**        | Map from product_id (string) to product text (`"Product: X. Aisle: Y. Department: Z."`).                                               |
| **eval_relevant_docs.json** | Map from query_id to list of product_ids that are in that order’s next basket (relevance labels for IR metrics).                       |
| **data_prep_params.json**   | Record of the data prep arguments and counts (n_train_pairs, n_eval_queries, n_corpus, etc.).                                          |

---

## Pipeline

### 1. Prepare

Reads the CSVs, builds one **anchor** (user context string) per target order, and for each product in that order’s “next” basket creates a **(anchor, positive)** pair with **positive** = product text. Splits orders into train vs eval (by order, not by pair), and writes the train/eval Datasets plus `eval_queries.json`, `eval_corpus.json`, `eval_relevant_docs.json` for the Information Retrieval evaluator.

**Key flags:** `--max-prior-orders`, `--max-product-names`, `--eval-frac`, `--output-dir`, `--data-dir`. Defaults: 5 prior orders, 20 product names, 10% eval. At the end the script prints the exact `--processed-dir` to use for training.

### 2. Train

Loads the processed dir (auto-resolves to a single param subdir under `processed/` if the default path has no `train_dataset`), builds a Sentence Transformer bi-encoder (default base: `all-MiniLM-L6-v2`), and trains with **MultipleNegativesRankingLoss** (in-batch negatives). Optionally runs **InformationRetrievalEvaluator** each epoch (Accuracy@k, MRR@10, NDCG@10, MAP@100). Saves checkpoints under `models/two_tower_sbert/` and, when IR eval is on, keeps the best by NDCG@10 in `models/two_tower_sbert/final/`.

**Key flags:** `--processed-dir`, `--output-dir`, `--lr`, `--epochs`, `--train-batch-size`, `--max-seq-length`, `--no-information-retrieval-evaluator` (faster runs, no IR metrics).

### 3. Serve

Loads the trained model from `final/` (or a checkpoint dir) and the product corpus from a JSON file. Encodes the corpus once at startup; for each query (user context string), encodes the query and returns the **top-k** product IDs by cosine similarity. Can be used from the CLI or via the Python API (`Recommender`, `recommend()`).

**Key flags:** `--model-dir`, `--corpus`, `--query` (raw context string), `--eval-query-id` (use a query from `eval_queries.json` by order_id), `--top-k`.

### Commands (copy-paste)

```bash
# 1. Prepare (writes to processed/p5_mp20_ef0.1/ with defaults)
uv run python -m src.data.prepare_instacart_sbert

# 2. Train (uses processed/p5_mp20_ef0.1 if it’s the only param subdir)
uv run python -m src.train.train_sbert --lr 1e-4
# Explicit dir: --processed-dir processed/p5_mp20_ef0.1
# Faster training (no IR eval): --no-information-retrieval-evaluator

# 3. Serve (demo: no --query uses built-in example)
uv run python -m src.inference.serve_recommendations --top-k 10
# With corpus and model: --corpus processed/p5_mp20_ef0.1/eval_corpus.json --model-dir models/two_tower_sbert/final
# Custom query: --query "[+7d w4h14] Milk, Bread."
# Real eval query: --eval-query-id <order_id>
```

---

## Prediction problem (summary)

- **Task:** Rank the catalog so products in the user’s _next_ order are at the top (see **What we are predicting** above).
- **Input:** User context from _prior_ orders only: a single text string built from the last N prior orders (product names in sequence, optional timing like “ordered 7 days after previous on weekday 4 at hour 14”). No information from the “next” order is included at prediction time.
- **Output:** A ranking over the full product catalog: each product gets a score (cosine similarity between the encoded context and the encoded product text). We return the top-k product IDs (and optionally scores).
- **Train vs serve:** For **training**, each (anchor, positive) pair has anchor = prior-only context (and during data prep we can optionally include “Next: weekday X, hour Y, …” in the anchor for that target order). The **positive** is one product that actually appears in that order’s next basket. For **serve and evaluation**, the query is the _same_ prior-only context **without** the “Next: …” segment, so we never use future information and the setup matches production.

---

## Results

### Data prep (example: max_prior_orders=5, max_product_names=20, eval_frac=0.1)

| Train pairs | Eval pairs | Eval queries | Corpus size |
| ----------- | ---------- | ------------ | ----------- |
| ~1.25M      | ~138k      | ~13k         | ~50k        |

Train/eval are split **by order** so that all pairs from a given order are in one split; eval queries are the hold-out orders, and the corpus is the full product set (~50k). Each eval query has one or more relevant products (the products actually in that order’s next basket).

### Example evaluation metrics

Setup: `processed/p5_mp20_ef0.1`, base model `all-MiniLM-L6-v2`, `max_seq_length` 256, default batch size and learning rate (e.g. `--lr 1e-4`). Evaluation runs over ~13k eval queries and ~50k corpus via the built-in `InformationRetrievalEvaluator`.

| Metric      | After 1 epoch | After 2 epochs | After 3 epochs | After 4 epochs |
| ----------- | ------------- | -------------- | -------------- | -------------- |
| Accuracy@1  | 0.210         | 0.226          | 0.239          | 0.239          |
| Accuracy@10 | 0.464         | 0.507          | 0.532          | 0.540          |
| Recall@10   | 0.103         | 0.116          | 0.125          | 0.129          |
| MRR@10      | 0.287         | 0.311          | 0.329          | 0.331          |
| NDCG@10     | 0.125         | 0.139          | 0.150          | 0.153          |
| MAP@100     | 0.071         | 0.078          | 0.085          | 0.086          |

**What the metrics mean:** Accuracy@k = fraction of queries where at least one relevant product appears in the top-k. Recall@10 = fraction of relevant products found in the top-10 (averaged per query). MRR@10 = mean reciprocal rank of the first relevant product in the top-10. NDCG@10 = normalized discounted cumulative gain at 10 (rewards relevant items ranked higher). MAP@100 = mean average precision over the top-100. All are computed per query and averaged; higher is better.

After one epoch the model puts at least one correct product in the top-10 for about **46%** of eval queries; after four epochs this reaches **~54%** (Accuracy@10). The trainer saves the best checkpoint by **NDCG@10** when the IR evaluator is enabled. Disable it with `--no-information-retrieval-evaluator` for faster training (validation loss only).

**Reproducibility:** Exact numbers depend on hardware, seed, and hyperparameters (e.g. batch size 64 vs 128, learning rate). Use the same data prep and train flags to approximate these results.

---

### Demo inference

The serve script can be run without `--query` to use a **built-in demo query** that mimics a user who previously ordered “Organic Milk, Whole Wheat Bread” in a context where the last order was 7 days prior, on weekday 4 at hour 14. The format is:

- **`[+7d w4h14]`** — shorthand for “ordered 7 days after previous order, on weekday 4 (0–6), at hour 14”.
- **`Organic Milk, Whole Wheat Bread.`** — product names from prior orders (sequence preserved, comma-separated).

So the full string is exactly what the data prep pipeline produces for the “anchor” side when we strip the “Next: …” part. You can pass any custom context with `--query "..."` or run on a stored eval query with `--eval-query-id <order_id>` (the script then loads that order’s context from `eval_queries.json`).

**Example run (no args beyond --top-k):**

```
[+7d w4h14] Organic Milk, Whole Wheat Bread.
```

**Example top-5 output:**

```
Top-5 recommendations:
  1. product_id=48628 (score=0.7877) Product: Organic Whole Wheat Bread. Aisle: bread. Department...
  2. product_id=13517 (score=0.7850) Product: Whole Wheat Bread. Aisle: bread. Department: bakery...
  3. product_id=44103 (score=0.7241) Product: Honey Whole Wheat Bread. Aisle: bread. Department: ...
  4. product_id=6454 (score=0.7207) Product: Whole Wheat Bread Loaf. Aisle: bread. Department: b...
  5. product_id=38591 (score=0.6624) Product: Organic Whole Wheat. Aisle: bread. Department: bake...
```

Scores are **cosine similarity** between the encoded query and each product embedding, in [0, 1] when embeddings are L2-normalized. Use `--eval-query-id` to test on a real eval order and compare with that order’s actual next basket.

---

## Training notes

- **Runtime:** Training is slow on CPU-only and on Apple Silicon (MPS): expect multiple hours for 5 epochs with ~1.2M pairs. Larger base models (e.g. `multi-qa-MiniLM-L6`) or longer `--max-seq-length` (e.g. 384 or 512) increase runtime and memory. Defaults (`all-MiniLM-L6-v2`, `max_seq_length` 256) are chosen for feasibility on a typical laptop or single GPU.
- **Hyperparameters:** Default learning rate `1e-4` and batch size (e.g. 64 or 128) work for many runs. The trainer uses a linear LR schedule with 10% warmup. Checkpoints are written every epoch under `models/two_tower_sbert/`; when the IR evaluator is on, the best checkpoint by NDCG@10 is also written to `models/two_tower_sbert/final/`.
- **Why ~1s+ per step (e.g. on Apple Silicon):** On MPS the code sets `dataloader_num_workers=0` and disables fp16 for stability. Data loading and tokenization run on the main thread, so the GPU often waits for the next batch and there is no prefetch. On **CUDA** you can pass `--dataloader-num-workers` (e.g. 4) for faster steps. With **dynamic padding** (pad to longest in batch), many steps are faster, but when the batch length (and thus tensor shape) changes, MPS may recompile and you can see occasional steps over 5s; **length bucketing** (fixed set of lengths) would limit recompilation to a few shapes.
- **Gradient accumulation:** Not used; each step is one batch. You can simulate a larger batch by increasing `--train-batch-size` if memory allows.

---

## Project structure

| Path                                       | Description                                                                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **data/**                                  | Raw Instacart CSVs (not in repo; user downloads from Kaggle).                                                                                     |
| **processed/<param>/**                     | Data prep output: `train_dataset/`, `eval_dataset/`, `eval_queries.json`, `eval_corpus.json`, `eval_relevant_docs.json`, `data_prep_params.json`. |
| **models/two_tower_sbert/**                | Training checkpoints (e.g. `checkpoint-58419/`) and `final/` (best by NDCG@10 when IR eval is on).                                                |
| **src/constants.py**                       | `PROJECT_ROOT`, `DEFAULT_DATA_DIR`, `DEFAULT_PROCESSED_DIR`, `DEFAULT_OUTPUT_DIR`, `DEFAULT_MODEL_DIR`, `DEFAULT_CORPUS_PATH`.                    |
| **src/utils.py**                           | `setup_colored_logging()`, `resolve_processed_dir()` (auto-resolve processed dir to a param subdir when needed).                                  |
| **src/data/prepare_instacart_sbert.py**    | Builds (anchor, positive) pairs from CSVs, splits train/eval by order, writes Datasets and IR artifacts.                                          |
| **src/train/train_sbert.py**               | Loads processed data, builds Sentence Transformer + MultipleNegativesRankingLoss, runs trainer with optional InformationRetrievalEvaluator.       |
| **src/inference/serve_recommendations.py** | Loads model and corpus, encodes query, returns top-k by cosine similarity; CLI and `Recommender` class.                                           |
| **notebooks/**                             | Jupyter notebooks for data prep, training, and serve (mirror the scripts for interactive use).                                                    |
| **pyproject.toml**, **uv.lock**            | Project and dependency lock (uv).                                                                                                                 |

---

## License

MIT. Use of the Instacart dataset is subject to its own terms (e.g. Kaggle).
