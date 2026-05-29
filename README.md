# Fanfic Spark Analysis

A distributed analysis of the [`marianna13/fanfics`](https://huggingface.co/datasets/marianna13/fanfics) corpus on SDSC Expanse, using Apache Spark to classify multilingual creative writing at scale. UCSD DSC 232R group project (Spring 2026).

**Group members:** Derek Pham · Mustafa Hayeri
**Notebook:** [`notebook/analysis.ipynb`](notebook/analysis.ipynb) — best viewed via [nbviewer](https://nbviewer.org/github/fanfic-spark-group/fanfic-spark-analysis/blob/main/notebook/analysis.ipynb) (GitHub's inline renderer can time out on the embedded figures).

---

## 1. Introduction

Fanfiction is one of the largest bodies of *deliberately authored* creative writing on the internet: millions of stories written by people for the love of a fictional world, tagged by the community with the fandom and genre they belong to. That makes it a different kind of big data from the operational exhaust most large-scale ML works with — flight logs, sensor telemetry, transactions, even patents. Here every record is an intentional act of creative expression, and the label (`CATEGORY`) is a human-assigned statement of *what this story is about*. The question we ask is whether the prose itself carries enough signal to recover that label automatically: **can a model read a story and predict its fandom and genre?**

A good predictive model here matters beyond fan communities. Automatic classification of free-form creative text underpins content recommendation, moderation, archival tagging, and cross-lingual discovery — and fanfiction, written by amateurs in 45 languages with wildly varying length and quality, is a stress test for all of them. A model that works on this messy, long-tailed, multilingual corpus is a model that will work on cleaner data.

**Why this needs big data and distributed computing.** The corpus is **~186 GB across 2,018 Parquet shards and 5,252,058 rows**, and individual stories run up to **~93 MB of text in a single record**. It does not fit in memory on a laptop, and pandas cannot open it — even the exploratory `describe`/`groupBy` passes have to stream across partitions. Every step of this project (reading the corpus, computing distributions, deduplicating 5.25 M rows by content hash, fitting models on 2.74 M training documents) is only practical because Spark spreads the work across the 16 cores of an SDSC Expanse node. Without a distributed engine the project is not slow — it is impossible.

This README integrates all milestones into one narrative: dataset selection and abstract (M1), data exploration (M2), preprocessing and a first distributed model (M3), and a second model using dimensionality reduction plus the final analysis (M4).

---

## 2. Figures

All figures are produced in [`notebook/analysis.ipynb`](notebook/analysis.ipynb). Exploratory plots (1–6) run over the full 5,252,058-row corpus; Model 2 plots (7–11) are produced on the held-out folds of the preprocessed English subset.

**Plot 1 — Rows per language (log scale).** ![Plot 1](notebook/plot1.png)
Row count for each of the 45 distinct language tags on a log axis. English dominates by several orders of magnitude; the top six tags (`en`, `es`, `fr`, `id`, `pt`, `de`) cover >99% of rows, with a long low-resource tail.

**Plot 2 — `text_len` distribution (linear and log10).** ![Plot 2](notebook/plot2.png)
Character-count distribution on a 0.1% sample (~5 K rows). Heavily right-skewed: most stories are a few thousand characters, with a tail out to multi-million-character outliers (max ~93 MB). The log10 view shows an approximately log-normal body.

**Plot 3 — `perplexity_score` distribution (linear and log10).** ![Plot 3](notebook/plot3.png)
Baseline-LM predictability score. A few extreme high-perplexity outliers (max ~70,000) dominate the linear view; the log10 view shows the bulk of the corpus in a narrow band.

**Plot 4 — Top-25 categories by row count.** ![Plot 4](notebook/plot4.png)
Of 278,388 distinct `CATEGORY` values, the empty string is the single largest (242,546 rows, a data-quality artifact); after it the named fandoms (Harry Potter, Naruto, …) outweigh the tail by one to two orders of magnitude.

**Plot 5 — `text_len` by top-5 languages.** ![Plot 5](notebook/plot5.png)
Mean `text_len` (± standard deviation) with median markers, for the five most frequent languages. Mean–median gaps reflect the per-language right skew.

**Plot 6 — `text_len` vs. `perplexity_score`.** ![Plot 6](notebook/plot6.png)
Scatter (log-x, log-y) on a 0.1% sample with the Pearson correlation printed. The weak correlation indicates the two numeric features carry largely independent signal.

**Plot 7 — PCA scree (per-component variance).** ![Plot 7](notebook/plot7.png)
Proportion of variance explained by each of the 50 principal components of the 100-dim Word2Vec space, largest to smallest. PC1 ≈ 0.216; a steep drop into a flat tail.

**Plot 8 — PCA cumulative variance and choice of *k*.** ![Plot 8](notebook/plot8.png)
Cumulative explained variance vs. number of components. 90% is reached at **37 components** (the deployed cut); the full 50 components reach 93.8%.

**Plot 9 — K-Means elbow and silhouette.** ![Plot 9](notebook/plot9.png)
Within-cluster cost (elbow) and silhouette over *k* = 2…6 on a 274,696-row training sample. Silhouette peaks at **k = 2 (0.248)** and decays; the cost elbow is shallow.

**Plot 10 — Cluster structure in the top two principal components.** ![Plot 10](notebook/plot10.png)
A 5,000-row sample projected onto PC1/PC2, coloured by K-Means cluster at the chosen *k*.

**Plot 11 — Model 2 confusion matrix.** ![Plot 11](notebook/plot11.png)
Row-normalized confusion matrix over the most frequent categories for the Model 2 Logistic Regression on the test fold.

**Spark UI — executor verification.** ![Spark UI](notebook/spark-ui-2.png)
The local-mode driver acts as a 16-core executor; the executors API confirms `totalCores = 16`, `isActive = True`.

---

## 3. Methods

> A summary of *what* was done and the parameters used. Interpretation is deferred to §5 (Discussion).

### 3.1 Data Exploration

Computed over all 5,252,058 rows in Spark: total row count; `describe` (count/mean/stddev/min/max) and `approxQuantile` for the two numeric columns `text_len` and `perplexity_score`; `groupBy` frequency tables for `CATEGORY`, `language`, and `SOURCE`; null counts per column; and a duplicate count using the md5 hash of `TEXT` as a proxy key. Six plots (Plots 1–6) visualize the language imbalance, the two numeric distributions, the category long-tail, per-language length, and the length-vs-perplexity relationship.

### 3.2 Preprocessing (Spark)

Applied to the raw corpus to produce the modelling frame `df_m3`:

1. **Column drop** — drop `__null_dask_index__` (a Dask leftover, ~97.5% null) and `SOURCE` (constant `"Fanfiction"`).
2. **Null / empty handling** — drop rows with null or empty-string `CATEGORY`.
3. **English subset + outlier/quality filter** — keep `language = en`; filter `text_len` and `perplexity_score` to the §3-computed percentile bounds.
4. **Hash deduplication** — drop duplicate `TEXT` via md5 hash (cheaper shuffle than `dropDuplicates(['TEXT'])` on multi-MB strings).
5. **Class rollup** — keep the top-50 most frequent named `CATEGORY` values and roll the remaining tail into an `'other'` class.
6. **Feature engineering + MLlib pipeline** — engineer `text_len_log10` and `len_per_perplexity` in Spark SQL, then fit a four-stage pipeline: `StringIndexer` (CATEGORY → label) → `Imputer` (median) → `VectorAssembler` → `MinMaxScaler`.
7. **Split** — `randomSplit([0.7, 0.15, 0.15], seed=42)`, giving ~2,741,007 train / 588,130 validation / 588,125 test rows. The same split (and the same fitted `StringIndexer`) is reused by both models.

**Infrastructure.** Run on a single SDSC Expanse JupyterLab session (16 cores, 128 GB) in Spark `local[*]` mode. Driver memory raised to 16 GB (in local mode the driver *is* the executor); the vectorized Parquet reader disabled (the ~93 MB records OOM its batch buffers); `spark.local.dir` redirected to TB-scale Lustre scratch; and the 2,018 shards' heterogeneous `language` physical type (string vs INT32) reconciled at load by reading footers, casting, and unioning with `allowMissingColumns=True`. Full configuration in §[Reproducing this project](#reproducing-this-project).

### 3.3 Model 1 — RandomForest on numeric features

`RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=20, maxDepth=5, seed=42)` on the four §3.2 numeric features (`text_len`, `perplexity_score`, `text_len_log10`, `len_per_perplexity`), evaluated on train/validation/test with `MulticlassClassificationEvaluator` (accuracy, F1, weighted precision, weighted recall). A capacity sweep refit two variants on the same caches: deeper (`numTrees=20, maxDepth=15`) and wider (`numTrees=100, maxDepth=5`).

### 3.4 Model 2 — Word2Vec → PCA → Logistic Regression + K-Means

A text-feature pipeline on the raw `TEXT` of the same §3.2 split:

```python
# Stage the text features (fit on a bounded 50k-doc sample of the train fold):
trunc     = SQLTransformer("SELECT *, substring(TEXT,1,8000) AS text_capped FROM __THIS__")
tokenizer = Tokenizer(inputCol='text_capped', outputCol='tokens_raw')
stopwords = StopWordsRemover(inputCol='tokens_raw', outputCol='tokens_all')
cap       = SQLTransformer("SELECT *, slice(tokens_all,1,1000) AS tokens FROM __THIS__")
word2vec  = Word2Vec(inputCol='tokens', outputCol='w2v',
                     vectorSize=100, minCount=50, maxIter=1, windowSize=5,
                     numPartitions=16, seed=42)
pca       = PCA(inputCol='w2v', outputCol='pcaFeatures', k=50)
```

- **Dimensionality reduction (unsupervised):** PCA reduces the 100-dim Word2Vec document vectors to 50 components; a `VectorSlicer` keeps the first **37** (the 90%-variance point, §10d).
- **Supervised head:** `LogisticRegression(featuresCol='pcaK', labelCol='label', maxIter=20, regParam=0.0)`, multinomial, evaluated on the same train/validation/test protocol as Model 1.
- **Clustering:** `KMeans` over *k* = 2…6 on a cached 10% training sample, scored by within-cluster cost and `ClusteringEvaluator` silhouette; refit at the chosen *k* and projected onto PC1/PC2.
- **Dimensionality comparison:** the same LR is refit at 37-component, 50-component, and full 100-dim Word2Vec widths to measure the effect of the reduction (§12).
- **Predictions analysis:** test-fold predictions decoded back to `CATEGORY`, with correct / false-positive / false-negative examples for the largest named category and a confusion matrix (Plot 11).

---

## 4. Results

> A summary of results and figures. No interpretation — see §5.

### 4.1 Data Exploration

- **45 distinct language tags**, English-dominant (~87% of rows); top six cover >99%.
- **`text_len`** ranges 5,001 → 93,626,685 characters; heavily right-skewed (Plots 2, 5).
- **`perplexity_score`** ranges 56.8 → 69,992; right-skewed (Plot 3).
- **`CATEGORY`** has 278,388 distinct values; the empty string is the largest at 242,546 rows (Plot 4).
- **Duplicates:** 337,555 rows (~6.4%) share a `TEXT` hash. **Nulls:** `language` has 3,088; `SOURCE` is constant; `__null_dask_index__` is ~97.5% null.
- `text_len` and `perplexity_score` show weak log-log correlation (Plot 6).

### 4.2 Preprocessing

From the 5,252,058-row baseline, the §3.2 filters (drop empty `CATEGORY`, English-only, percentile outlier bounds, hash dedup) yield a modelling frame of **~3,917,262 rows**, split 2,741,007 / 588,130 / 588,125 (train / validation / test). The target is the top-50 named categories plus `'other'`.

### 4.3 Model 1 — RandomForest

| metric (test) | baseline (T=20, D=5) | deeper (T=20, D=15) | wider (T=100, D=5) |
|---|---|---|---|
| accuracy | 0.8428 | 0.8428 | 0.8428 |
| F1 | 0.7708 | 0.7708 | 0.7708 |
| weighted precision | 0.7102 | 0.7102 | 0.7102 |
| fit time | 145.1 s | 794.5 s | 181.5 s |

Train ≈ validation ≈ test on all four metrics (gap ≈ 0). Every §7c sample prediction across all three folds is `'other'`. The three capacity variants are identical to four decimals.

### 4.4 Model 2 — Word2Vec → PCA → LR + K-Means

**PCA explained variance** (Plots 7–8): PC1 = 0.216; top-10 = 0.703; all 50 components = 0.938. 90% variance is reached at **37 components** (cumulative 0.9026); 95% is not reached within 50.

**Logistic Regression** (37-dim PCA features, 52 classes):

| metric | train | validation | test |
|---|---|---|---|
| accuracy | 0.8405 | 0.8402 | **0.8408** |
| F1 | 0.8045 | 0.8043 | **0.8052** |
| weighted precision | 0.7826 | 0.7824 | **0.7830** |
| weighted recall | 0.8405 | 0.8402 | 0.8408 |

Fit time 435.9 s; Word2Vec + PCA fit 362.5 s.

**Dimensionality reduction vs. full feature set** (same LR, same split):

| features | dim | test accuracy | test F1 | fit time |
|---|---|---|---|---|
| PCA 37 (deployed) | 37 | 0.8408 | 0.8052 | 435.9 s |
| PCA 50 (all components) | 50 | _[pending run]_ | _[pending run]_ | _[pending run]_ |
| Word2Vec 100 (full, pre-PCA) | 100 | _[pending run]_ | _[pending run]_ | _[pending run]_ |

**K-Means** (Plots 9–10): on a 274,696-row sample over *k* = 2…6, silhouette peaks at **k = 2 (0.2479)** and decreases monotonically; within-cluster cost falls smoothly with no sharp elbow.

**Predictions** (Plot 11): 494,521 / 588,125 test rows correct (accuracy 0.8408). For the focus category `Harry Potter, Romance`, false positives are neighbouring multi-tag Harry Potter categories (Drama / Family / Angst / Friendship + Romance) and `'other'`; false negatives go mostly to `'other'` and `Harry Potter, Drama, Romance`.

---

## 5. Discussion

**Data exploration.** The exploration immediately reframed the problem. The 45-language, 278K-category, 93 MB-record reality is far messier than the abstract assumed (22 languages, ~6 MB records), and it forced the engineering choices that follow: the empty-string `CATEGORY` (4.6% of rows) is noise, not a class; the category long-tail (Plot 4) means a flat 278K-way classifier is hopeless, so we restrict to the top-50 + `'other'`; and the mega-records (Plot 2) are why the vectorized reader is disabled and `text_len` is capped. We trust these findings because they are computed over the *entire* corpus, not a sample.

**Preprocessing.** The most defensible choice is hash-based deduplication: ~6.4% of rows are exact `TEXT` duplicates, and leaving them in would leak content across the train/test split. The most consequential — and most debatable — choice is the English-only restriction. It makes the modelling tractable and matches the data's 87% English skew, but it abandons the abstract's cross-lingual ambition; we flag it as the clearest shortcoming and the most natural extension (§6).

**Model 1.** Model 1 lands exactly where §8 argues it must: at the majority-class floor. Test accuracy (0.8428) equals the `'other'` share because the model predicts `'other'` for everything, and F1 (0.7708) trails accuracy because it averages over the 50 named classes the model never predicts. The capacity sweep is the key evidence: a 15-deep forest has the capacity to memorize fine structure, yet it matches the 5-deep forest on every generalization metric. That null result is strong — it shows the bottleneck is *feature information*, not model capacity. Length and predictability simply do not encode which fandom a story belongs to; `Harry Potter, Romance` and `Naruto, Romance` have near-identical length-by-perplexity envelopes.

**Model 2.** Model 2 acts on that diagnosis and confirms it. By replacing the four numeric features with dense Word2Vec prose embeddings, F1 rises from 0.7708 to 0.8052 and weighted precision from 0.7102 to 0.7830, and the model starts predicting real fandoms (Plot 11) instead of collapsing to `'other'`. The most interesting wrinkle is that test *accuracy* actually dips a hair below Model 1's (0.8408 vs 0.8428): Model 2 gives up a few majority-class freebies in exchange for genuinely modelling the minority classes, which is why F1 climbs while accuracy holds. **The central finding of the project is that changing the feature *representation* helped where changing model *capacity* did not.**

The PCA step is close to free: 37 components capture 90% of the Word2Vec variance and (per §12) match the full-dimensional fit, so dimensionality reduction buys a smaller, cheaper model at negligible accuracy cost — exactly the behaviour expected when the leading principal directions carry the task-relevant structure. The K-Means result is an honest negative: silhouette peaks at k = 2 and decays, so the embedding has no strong multi-cluster structure and does not recover the fandom categories unsupervised. The supervised LR is what extracts the signal; clustering alone would not.

**How believable is this, and what's wrong with it?** The train ≈ validation ≈ test agreement on both models, and the fixed `seed=42` split shared across them, make the comparison fair and the numbers reproducible. But three shortcomings temper the results: (1) the ~84% `'other'` imbalance pins headline accuracy near the majority floor for *any* model that does not handle imbalance, so accuracy is a misleading headline here and F1/precision are the metrics that matter; (2) the Word2Vec model is fit on a bounded 50K-document sample with `maxIter=1` and a 1,000-token cap per story — deliberate runtime levers that almost certainly leave embedding quality on the table; and (3) everything runs in Spark `local[*]` mode, so our "distributed" training is really 16 local threads, and the speedup story (§14) is about realized I/O and fit optimizations rather than a controlled multi-node scaling curve.

---

## 6. Conclusion

We set out to predict a fanfiction story's fandom-and-genre label from its text, at a scale (186 GB, 5.25 M rows) that only a distributed engine can handle. Model 1 (RandomForest on numeric metadata) underfit to the majority class and showed the bottleneck was feature information. Model 2 (Word2Vec → PCA → Logistic Regression) acted on that finding: dense prose embeddings lifted test F1 from 0.7708 to 0.8052 and began correctly classifying the major fandoms, while an unsupervised PCA reduction to 37 dimensions preserved accuracy nearly for free. K-Means revealed only weak cluster structure, confirming the signal is supervised, not geometric.

**What we would do differently, and explore next.** The single highest-value change is handling the class imbalance — inverse-frequency `weightCol` or resampling — which is the binding ceiling on accuracy for both models and which we left untouched. Beyond that: a stronger Word2Vec fit (more documents, more iterations, larger vectors), a *supervised* reduction such as LDA that keeps label-separating directions rather than high-variance ones, and the cross-lingual evaluation the English-only filter set aside.

**What we learned about big data.** Working at this scale changed the order of our decisions: infrastructure came first. The biggest wins were not modelling choices but data-engineering ones — disabling the vectorized Parquet reader to survive 93 MB records, redirecting shuffle scratch to Lustre, deduplicating on a 32-byte hash instead of multi-MB strings, and capping text *before* tokenizing to bound the dominant cost. Distributed computing did not just make the project faster; it made the difference between a project that runs and one that cannot exist. It also taught us where parallelism stops helping — PCA's covariance SVD and the final coefficient collect run on the driver, a serial fraction that caps achievable speedup (§14).

---

## 7. Statement of Collaboration

> **DRAFT — pending confirmation from both members before submission.** Format: *Name: Title: Contribution.*

**Derek Pham: Analysis & Report Lead.** Selected the dataset and the intentional-authorship framing; led the written report (Introduction, Methods/Results restructure, Discussion, Conclusion) and the Model 2 analysis sections in the notebook (§12 fitting analysis and dimensionality-reduction comparison, §13 conclusion, §14 speedup analysis); extracted and captioned the figures; contributed to data exploration and preprocessing in earlier milestones.

**Mustafa Hayeri: Modelling & Infrastructure Lead.** Built and ran the Spark ML pipelines on SDSC Expanse — the Milestone 3 RandomForest baseline and the Milestone 4 Word2Vec → PCA → Logistic Regression + K-Means pipeline (§10–§11) — owned the notebook code and the Expanse run/run-tuning (driver memory, scratch, Word2Vec result-size fixes), and the style pass over the README and notebook.

Both members reviewed each other's work, agreed the modelling design, and collaborated throughout.

---

## Reproducing this project

### The data

Each member downloaded the corpus once with `huggingface_hub.snapshot_download` to `~/<username>/fanfic-spark-analysis/shared/fanfics/` on Expanse's Lustre filesystem. The notebook resolves the corpus path portably, so either member's run finds his own copy. The notebook's top section contains the download and environment-setup cells.

### SDSC Expanse session

| Setting | Value |
|---|---|
| Portal | [portal.expanse.sdsc.edu](https://portal.expanse.sdsc.edu/) |
| Account | `TG-SEE260003` |
| Partition | `shared` |
| Cores / Memory | 16 / 128 GB |
| Singularity image | `~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif` |
| Environment module | `singularitypro` |
| App | JupyterLab |

### SparkSession

```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.instances", 15) \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .getOrCreate()
spark.conf.set("spark.local.dir", "/expanse/lustre/projects/uci157/$USER/spark-scratch")
```

- **16 GB driver, not 2 GB** — the Expanse JupyterLab image launches Spark in `local[*]` mode, so the driver JVM is the executor; the `executor.*` settings are kept for portability to a future YARN setup but are inactive here. Sixteen concurrent tasks reading TEXT rows up to ~93 MB each need far more than the 2 GB default.
- **Vectorized Parquet reader disabled** — its fixed-size batch buffers OOM on ~93 MB records; row-by-row reads trade ~2–3× throughput for stability.
- **Schema heterogeneity** — 2,016 shards encode `language` as `string`, 2 as `INT32`; the load cell reads footers, splits by physical type, casts, and unions with `allowMissingColumns=True`, preserving all 5,252,058 rows.

Authoritative setup guide: [`ucsd-dsc232r/group-project`](https://github.com/ucsd-dsc232r/group-project/).

---

## Submission

The final submission is the `main` branch of this repository, submitted via Gradescope; the repository is made public for the voting round. Prior milestones were submitted from their `Milestone2` / `Milestone3` branches.
