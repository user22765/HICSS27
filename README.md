# HICSS27 submission

This repository contains the code and supplementary materials for the paper *"From PDFs to machine-readable theory: multimodal extraction of tested construct relations in SEM-based research."*

## Repository structure

- [`code/`](code) — extraction and evaluation scripts
- [`prompts/`](prompts) — prompts used for the extraction models
- [`images/`](images) — figures, including per-paper score histograms

## Code

The extraction and evaluation scripts can be found in the [`code/`](code) directory. The prompts used for the extraction models are in the [`prompts/`](prompts) directory.

## Manual evaluation

The output of the manual evaluation for a fixed run of the combined results is available in [`here`](code/evaluation/combined_equivalents_pubmed.json).

In this file, keys are ground-truth construct names and values are the corresponding extracted construct names.

## Distribution of per-paper scores

The paper reports micro F1 scores, i.e., extractions are pooled across all papers before the scores are calculated. As a supplement, we also provide the distribution of the combined results as histograms of macro F1 scores, where scores are calculated per paper and then averaged.

The histogram images can be found in [`images/histograms`](images/histograms).
