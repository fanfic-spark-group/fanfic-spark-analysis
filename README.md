# Fanfic Spark Analysis

A distributed analysis of the [`marianna13/fanfics`](https://huggingface.co/datasets/marianna13/fanfics) corpus on SDSC Expanse, using Apache Spark to characterize multilingual creative writing at scale. UCSD DSC 232R group project.

## Group Members

- Derek Pham
- Mustafa Hayeri

## Dataset

- **Source:** [`marianna13/fanfics` on HuggingFace](https://huggingface.co/datasets/marianna13/fanfics)
- **Size:** ~186 GB across 2,018 Parquet shards
- **Records:** 5,252,058 rows
- **Languages:** 45 distinct values, English-dominant (~87%); top six (`en`, `es`, `fr`, `id`, `pt`, `de`) cover >99% of rows
- **Schema (7 columns):**
  - `__null_dask_index__` (long) — Dask index leftover; present in only ~2.5% of shards (will be dropped)
  - `TEXT` (string) — full prose; observed `text_len` ranges from 5,001 to 93,626,685 characters (max ~93 MB per record)
  - `CATEGORY` (string) — fandom + genre label, ~278 K distinct values; target column for downstream classification
  - `SOURCE` (string) — verified constant per record (`"Fanfiction"` for all 5.25 M rows; will be dropped in preprocessing)
  - `language` (string) — pre-classified language tag, 45 distinct values plus 3,088 NULLs
  - `text_len` (long) — character count of `TEXT`
  - `perplexity_score` (double) — baseline-LM predictability score, observed range 56.8–69,992

## SDSC Expanse Setup

All work was performed on SDSC Expanse via the JupyterHub portal at [portal.expanse.sdsc.edu](https://portal.expanse.sdsc.edu/).

**Jupyter session configuration:**

| Setting | Value |
|---|---|
| Account | `TG-SEE260003` |
| Partition | `shared` |
| Cores | 16 |
| Memory | 128 GB |
| Singularity image | `~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif` |
| Environment module | `singularitypro` |
| App type | JupyterLab |

**First-time setup** (per the [`ucsd-dsc232r/group-project` README](https://github.com/ucsd-dsc232r/group-project/)): each group member created the standard symbolic links into `/expanse/lustre/projects/uci157/` for the user's personal folder, the shared singularity images folder, and group-member folders.

**Dataset staging:** each group member downloaded the corpus once via `huggingface_hub.snapshot_download` to `~/<username>/fanfic-spark-analysis/shared/fanfics/` on Lustre. The notebook resolves the corpus path portably so either member's run finds his own copy without edits.

## SparkSession Configuration

We allocated **16 cores / 128 GB** in the Jupyter session. The formula from [`ucsd-dsc232r/group-project/SPARK_HPC_BEST_PRACTICES.md`](https://github.com/ucsd-dsc232r/group-project/blob/main/SPARK_HPC_BEST_PRACTICES.md) is the starting point, with two corpus-specific adjustments:

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

**Spark scratch redirected to Lustre.** The default `spark.local.dir` is `/tmp` on the compute node, which is small (a few GB) and exhausts under any TEXT-keyed shuffle on this corpus. The notebook sets `spark.local.dir` to `/expanse/lustre/projects/uci157/$USER/spark-scratch` (TB-scale Lustre space) at SparkSession build time so spills land in the user's project directory.

**Why 16 GB driver (not 2 GB)?** The Expanse JupyterLab Spark image launches in `local[*]` mode rather than provisioning YARN executors. In local mode the driver JVM **is** the executor — `executor.memory` and `executor.instances` are inactive and all task memory pressure falls on the driver heap. Sixteen concurrent tasks reading TEXT rows up to ~93 MB each (the max observed `text_len` is 93,626,685 characters) need substantially more than the 2 GB default. We bump driver memory to 16 GB; the documented `executor.*` settings stay in the builder for portability to a future YARN setup but are no-ops here.

**Why disable the vectorized Parquet reader?** The corpus contains TEXT records up to ~93 MB. Spark's default vectorized Parquet reader allocates contiguous on-heap buffers proportional to its batch size (4,096 rows) and column width, which OOMs the JVM when batches contain large TEXT values. Disabling vectorized reading switches to row-by-row reads with bounded per-row memory; we accept a small read-throughput cost (~2-3×) in exchange for stability across the corpus.

**Schema-heterogeneity note.** The corpus's 2,018 Parquet shards do not share a uniform schema: 2,016 shards encode `language` as `string` while 2 encode it as `INT32`, and most shards omit `__null_dask_index__` entirely (only ~2.5% of rows carry it). The notebook's data-load cell reads each shard's Parquet footer, splits the file list by `language` physical type, loads the two subsets, casts the INT32 subset to string, and unions them with `allowMissingColumns=True`. This preserves all 5,252,058 rows; the heterogeneity itself surfaces in the §3 frequency tables and is addressed in the §5 preprocessing plan.

**Spark UI** (SparkSession HTML widget plus full configuration printout — local-mode driver acting as a 16-core executor; the executors-API output in cell 17 of the notebook confirms `totalCores=16`, `isActive=True`):

![Spark UI](notebook/spark-ui.png)

## Notebook

Full data exploration is in [`notebook/analysis.ipynb`](notebook/analysis.ipynb), covering:

1. SparkSession initialization and configuration verification
2. Dataset loading from Lustre and partition inspection
3. Data exploration via Spark DataFrames (count, schema, describe, group-by aggregations, distinct counts, missing/duplicate analysis)
4. Distribution plots (language frequency, text-length distribution, perplexity distribution, top categories, text-length × language, text-length × perplexity scatter)
5. Preprocessing plan for Milestone 3

## Submission

Final submission for Milestone 2 is the `Milestone2` branch URL of this repository, submitted via Gradescope.
