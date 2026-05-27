# Fanfic Spark Analysis

A distributed analysis of the [`marianna13/fanfics`](https://huggingface.co/datasets/marianna13/fanfics) corpus on SDSC Expanse, using Apache Spark to study multilingual creative writing at scale. UCSD DSC 232R group project.

## Group members

- Derek Pham
- Mustafa Hayeri

## Dataset

- **Source:** [`marianna13/fanfics` on HuggingFace](https://huggingface.co/datasets/marianna13/fanfics)
- **Size:** ~186 GB across 2,018 Parquet shards
- **Records:** 5,252,058 rows
- **Languages:** 45 distinct values, English-dominant (~87%); the top six (`en`, `es`, `fr`, `id`, `pt`, `de`) cover over 99% of rows
- **Schema (7 columns):**
  - `__null_dask_index__` (long): Dask index leftover, present in only ~2.5% of shards (dropped in preprocessing)
  - `TEXT` (string): full prose; observed `text_len` ranges from 5,001 to 93,626,685 characters (max ~93 MB per record)
  - `CATEGORY` (string): fandom and genre label, ~278K distinct values; the target column for downstream classification
  - `SOURCE` (string): constant per record (`"Fanfiction"` for all 5.25M rows; dropped in preprocessing)
  - `language` (string): pre-classified language tag, 45 distinct values plus 3,088 NULLs
  - `text_len` (long): character count of `TEXT`
  - `perplexity_score` (double): baseline-LM predictability score, observed range 56.8 to 69,992

## SDSC Expanse setup

We ran everything on SDSC Expanse through the JupyterHub portal at [portal.expanse.sdsc.edu](https://portal.expanse.sdsc.edu/).

Jupyter session settings:

| Setting | Value |
|---|---|
| Account | `TG-SEE260003` |
| Partition | `shared` |
| Cores | 16 |
| Memory | 128 GB |
| Singularity image | `~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif` |
| Environment module | `singularitypro` |
| App type | JupyterLab |

Following the [`ucsd-dsc232r/group-project` README](https://github.com/ucsd-dsc232r/group-project/), each group member created the standard symbolic links into `/expanse/lustre/projects/uci157/`: one for the personal folder, one for the shared singularity-images folder, and one for the group-member folders.

Each group member downloaded the corpus once with `huggingface_hub.snapshot_download` to `~/<username>/fanfic-spark-analysis/shared/fanfics/` on Lustre. The notebook resolves the corpus path portably, so either member's run finds his own copy without edits.

## SparkSession configuration

We allocated 16 cores and 128 GB in the Jupyter session. The formula in [`ucsd-dsc232r/group-project/SPARK_HPC_BEST_PRACTICES.md`](https://github.com/ucsd-dsc232r/group-project/blob/main/SPARK_HPC_BEST_PRACTICES.md) is the starting point, with two corpus-specific adjustments:

```
Driver memory       = 16 GB (local-mode override, see below)
Executor instances  = Total Cores - 1 = 15
Executor memory     = (Total Memory - Driver Memory) / Executor Instances
                    = (128 - 16) / 15 ≈ 7.5 GB → kept at 8 GB for documentation
```

Resulting builder:

```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.instances", 15) \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .getOrCreate()
```

### Spark scratch on Lustre

The default `spark.local.dir` is `/tmp` on the compute node, which is small (a few GB) and runs out under any TEXT-keyed shuffle on this corpus. The notebook sets `spark.local.dir` to `/expanse/lustre/projects/uci157/$USER/spark-scratch` (TB-scale Lustre space) at SparkSession build time, so spills land in the user's project directory.

### Why 16 GB driver, not 2 GB

The Expanse JupyterLab Spark image launches in `local[*]` mode rather than provisioning YARN executors. In local mode the driver JVM is the executor: `executor.memory` and `executor.instances` are inactive, and all task memory pressure falls on the driver heap. Sixteen concurrent tasks reading TEXT rows up to ~93 MB each (the maximum observed `text_len` is 93,626,685 characters) need much more than the 2 GB default. We raise driver memory to 16 GB. The documented `executor.*` settings stay in the builder for portability to a future YARN setup but do nothing here.

### Why disable the vectorized Parquet reader

The corpus contains TEXT records up to ~93 MB. Spark's default vectorized Parquet reader allocates contiguous on-heap buffers proportional to its batch size (4,096 rows) and column width, which OOMs the JVM when batches contain large TEXT values. Disabling vectorized reading switches to row-by-row reads with bounded per-row memory. We accept a small read-throughput cost (roughly 2 to 3 times slower) in exchange for stability across the corpus.

### Schema heterogeneity

The 2,018 Parquet shards do not share a uniform schema: 2,016 shards encode `language` as `string` while 2 encode it as `INT32`, and most shards omit `__null_dask_index__` entirely (only ~2.5% of rows carry it). The notebook's data-load cell reads each shard's Parquet footer, splits the file list by `language` physical type, loads the two subsets, casts the INT32 subset to string, and unions them with `allowMissingColumns=True`. This preserves all 5,252,058 rows. The heterogeneity shows up in the §3 frequency tables and is handled in the §5 preprocessing plan.

Spark UI (the SparkSession HTML widget plus the full configuration printout). The local-mode driver acts as a 16-core executor; the executors-API output in cell 17 of the notebook confirms `totalCores=16` and `isActive=True`:

![Spark UI](notebook/spark-ui-2.png)

## Exploratory data analysis

Six plots over the full 5,252,058-row corpus. The full code and the §3 `describe` / `groupBy` / missing / duplicate analysis live in [`notebook/analysis.ipynb`](notebook/analysis.ipynb); the most important findings are below.

### Plot 1: rows per language

![Plot 1: rows per language](notebook/plot1.png)

A log-scale bar chart over all 45 distinct language tags shows the scale of the language imbalance. English has many orders of magnitude more rows than the smallest tier. This imbalance shapes the §5 preprocessing plan: any cross-lingual transfer evaluation has to account for low-resource languages, and per-language stratified sampling has to handle the long tail. The 45 distinct tags are more than the 22 the abstract assumed; §3 records the corrected count.

### Plot 2: text_len distribution

![Plot 2: text_len distribution](notebook/plot2.png)

The histogram is heavily right-skewed. Most records sit in the few-thousand-character range, with a long tail out to multi-million-character outliers. The linear-x view is unreadable on its own, so a log10-x version sits next to it, and that version shows an approximately log-normal body. From the §3 quantiles, the largest single-record `TEXT` reaches ~93 MB (about 93 million characters), far larger than the ~6 MB the abstract assumed. These mega-records drive the §5 preprocessing decisions: cap or filter `text_len` at the 99th percentile to prevent partition skew during downstream model training, and keep the vectorized Parquet reader disabled because of these outliers.

### Plot 3: perplexity_score distribution

![Plot 3: perplexity_score distribution](notebook/plot3.png)

Perplexity measures how predictable each record is under a baseline language model. The linear-x view is dominated by a few extreme high-perplexity outliers (the §3 maximum was ~70,000), so a log10-x version sits next to it, and that version shows the bulk of the corpus in a much narrower predictability band. The shape of this distribution sets the §5 quality-filter threshold: extreme high-perplexity records (gibberish, broken encodings, non-natural-language artifacts) can be excluded, and extreme low-perplexity records (boilerplate, copy-paste artifacts) likely carry little useful signal.

### Plot 4: top-25 categories

![Plot 4: top-25 categories](notebook/plot4.png)

The target column is heavily long-tailed. Of 278,388 distinct `CATEGORY` values, the top entry is the empty string (242,546 rows), which is a data-quality artifact rather than a real category. After that, the named fandoms (Harry Potter, Naruto, and so on) outweigh the rest of the tail by one to two orders of magnitude. For Milestone 3 we (a) drop empty-`CATEGORY` rows entirely, then (b) restrict the classifier to the top-K most frequent named categories (K ≈ 50, refined from this plot) and roll the remaining tail into an `"other"` class. This keeps the problem tractable and class balance manageable.

### Plot 5: text_len by top-5 languages

![Plot 5: text_len by top-5 languages](notebook/plot5.png)

The bars show mean `text_len` with standard-deviation error bars; the red dots mark the median. The gap between mean and median reflects the right-skew in each language's record-length distribution, since a few mega-records pull the means well above the medians. The differences across languages point to writing-style or platform variation between language communities, which matters for the cross-lingual transfer plan in §5: a model trained on long English texts may generalize poorly to a community that writes shorter pieces, and outlier-driven means call for median-based filtering rather than mean-based filtering.

### Plot 6: text_len vs perplexity_score

![Plot 6: text_len vs perplexity_score](notebook/plot6.png)

This scatter (log-x for `text_len`, log-y for `perplexity_score`) shows whether longer texts are systematically more or less predictable. Low correlation means `text_len` and `perplexity_score` carry distinct signal, so both are worth keeping as features. Strong correlation would mean redundancy and let us drop one in §5 preprocessing. The printed Pearson correlation on the log-log values quantifies this. We expect a weak negative correlation (longer texts tend to be slightly more predictable for a baseline LM), but small enough that both features stay.

## Key takeaways

- 45 distinct languages, English-dominant (~87%). The top six (`en`, `es`, `fr`, `id`, `pt`, `de`) cover over 99% of rows; the long tail spans many orders of magnitude. The Milestone 3 baseline restricts to the English subset; Milestone 4 may explore a multilingual variant with per-language stratified sampling.
- Mega-records up to ~93 MB drive the infrastructure choices. We disabled the vectorized Parquet reader, raised driver memory to 16 GB, redirected Spark scratch to Lustre, and cap `text_len` at the 99th percentile in preprocessing.
- `CATEGORY` is the noisiest column. 242,546 rows (~4.6%) have an empty-string `CATEGORY`, which we drop unconditionally as a data-quality artifact rather than a real label. The 278,388 distinct values are too many for a flat classifier, so we restrict to the top-K (K ≈ 50) named categories with an `"other"` bucket.
- About 6.4% of `TEXT` rows are duplicates (337,555 rows, detected with an md5-hash proxy in §3). We de-duplicate before training using the same hash pattern, which is cheaper than `dropDuplicates(['TEXT'])` because the shuffle runs on a 32-byte hash instead of a multi-MB string.
- `text_len` and `perplexity_score` carry distinct signal. Plot 6 shows weak log-log correlation, so both stay as features.
- `SOURCE` is constant (1 distinct value) and `__null_dask_index__` is 97.5% null. We drop both unconditionally in preprocessing.
- The Milestone 3 baseline hits the majority-class floor. `RandomForestClassifier(numTrees=20, maxDepth=5)` on the §6f numeric pipeline (`text_len`, `perplexity_score`, engineered `text_len_log10` and `len_per_perplexity`, median-imputed and min-max scaled) reaches test accuracy near the §6e `'other'` share (~0.842), and every §7c sample across train, validation, and test is predicted `'other'`. Train, validation, and test match on all four metrics (gap ≈ 0), which is a severe underfit rather than overfit.
- The Milestone 3 capacity sweep was not the bottleneck. The §8b deeper (T=20, D=15) and wider (T=100, D=5) variants produce metrics identical to the baseline and never open a train-test gap; the deeper variant costs several times the baseline's fit time for no metric gain. Even with the §6f engineered features and scaling, the numeric feature space cannot separate the 51 fandom-genre classes regardless of model capacity. The bottleneck is feature information, not classifier capacity.
- Milestone 4 plan. Three candidates are queued in §8d: (1) `LogisticRegression(multinomial)` on `HashingTF` + `IDF`, the §5 original-plan classifier that the §6f pivot deferred, since linear models tolerate the 2^18-dim sparse space natively; (2) `Word2Vec` embeddings plus LR, a dense alternative to TF-IDF for the same target; (3) a cross-lingual variant that lifts the English-only filter and evaluates per-language transfer on the top six languages from §3.

## Notebook

The full analysis is in [`notebook/analysis.ipynb`](notebook/analysis.ipynb). It covers SparkSession initialization, dataset loading from Lustre and partition inspection, the §3 data exploration (`describe`, `groupBy`, distinct counts, missing and duplicate analysis), the six §4 plots with their analysis cells, the §5 written preprocessing plan, and the §6 preprocessing implementation. The §6 work drops dead columns, handles null and empty-`CATEGORY` rows, takes the English subset with a percentile outlier filter, deduplicates by hash, rolls categories into the top 50 plus an `'other'` bucket, checkpoints to Parquet, engineers the `text_len_log10` and `len_per_perplexity` features in Spark SQL, and fits a four-stage MLlib pipeline (`StringIndexer` → `Imputer` → `VectorAssembler` → `MinMaxScaler`) that covers all four Part 1 transformer categories, ending with the train/validation/test split. §7 trains the baseline `RandomForestClassifier` with train/validation/test evaluation, per-fold sample predictions, and executor verification (with an honest local-mode caveat). §8 is the fitting analysis (Part 3): fitting-graph placement, two RF hyperparameter variants compared side by side, a best-model discussion, and the Milestone 4 model candidates. §9 is the conclusion: the model verdict, improvement paths, and the role of distributed computing.

## Submission

The final submission for Milestone 3 is the `Milestone3` branch URL of this repository, submitted via Gradescope.
