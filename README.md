# Supplementary material

Code, prompts, model generations, and judge outputs for
"Representing and Steering a Hypothesized Consciousness Spectrum: Manifold-Guided vs. Linear Directions".

The annotated dataset (636 sentences with tiers and scores) will be released upon acceptance.
Scripts that read `full_dataset_with_scores_new.csv` or sentence embeddings therefore need the dataset to run;
all outputs derived from it are included.

## Code

| Paper section | Scripts |
| --- | --- |
| Embeddings, probing, ablation (2.1, 3.1, App. E) | `1_get_sentences_embeddings.py`, `2_ablation.py`, `2b_ablation_llm.py`, `2c_tier_probe_llm.py` |
| Permutation test and TF-IDF baseline (App. E.3, E.4) | `appE3_permutation_test.py`, `appE4_tfidf_baseline.py` |
| Semantic confounds (App. C.1) | `appC1_confound_directions.py`, `appC1_valence_vader_vs_nrcvad.py` |
| Alternative labelings (App. C.2) | `appC2_alternative_labeling_geometry.py` |
| Writing-style confounds (App. C.3) | `appC3_writing_style_confound.py` |
| kNN graph, UMAP, TTS, GES, containment (2.2-2.3, 3.2, App. C, H-K) | `3_build_knn.py`, `4_knn_umap.py`, `5_tts.py`, `5b_tts_ges_llm.py`, `6_ges_with_heatmap.py`, `7_convex.py` |
| Dimensionality diagnostics (App. D, G) | `manifold_structure_study/` |
| Utility-guided and geometry-only trajectories (2.4, 3.3, App. F, L) | `8_gradient_guide_path.py`, `9_no_revisit_greedy_walk.py`, `10_mark_path.py` |
| Layer selection (2.5.1, App. M) | `11_get_llm_activations.py`, `12_select_steering_layer.py`, `20c_plot_layer_selection.py` |
| Steering (2.5.2) | `13_build_steering_vectors.py`, `14_steering_hooks.py`, `15_generate_with_steering.py`, `run_steering_pipeline.py`, `model_config.py`, `model_runtime.py` |
| Short-form factuality (TriviaQA) | `17b_fetch_triviaqa.py`, `17b_generate_triviaqa.py`, `17c_score_triviaqa.py` |
| Long-form factuality | `17_generate_factuality.py`, `18_llm_judge.py`, `20b_plot_multimodel_factuality.py` |
| Generation quality (pairwise) | `16_generate_mental_health.py`, `32_build_pairs_plan.py`, `18_llm_judge.py --task mental_health_pairwise`, `34b_plot_multimodel_pairwise.py` |
| Annotation reliability (App. B.5) | `dataset_agreement_test/` |

Install dependencies with `pip install -r requirements.txt`.
App. C scripts additionally need `vaderSentiment`, the NRC VAD lexicon (v2.1), the Brysbaert et al. (2014) concreteness norms, and the `s-nlp/roberta-base-formality-ranker` classifier; the lexicons are not redistributed here.

## Prompts

- `triviaqa_prompts.json`: the 500 TriviaQA questions (rc.nocontext validation, fixed-seed subsample).
- `longform_factuality_prompts.json`: the 25 open-ended long-form questions.
- `mh_diag_prompts.json`: prompts targeting low-level tiers used for the pairwise evaluation.
- `llm_pair_wise_rubric.txt`: the pairwise judging rubric; judgments in `all_mh_judged/` were collected by giving this rubric and the `to_judge` sheet to the judge in a chat interface, and `18_llm_judge.py --task mental_health_pairwise` applies the same rubric through an API.
- `dataset_agreement_test/part*/rubric_*.txt`: rubrics given to the blind LLM annotators.

## Results

- `all_long_form_generation/`, `all_long_form_judged/`: long-form generations and claim-level judgments for all six LLMs.
- `all_triviaqa/`: TriviaQA accuracy per condition and alpha; per-question generations are in
  `hf_results/*/factuality_triviaqa_*.jsonl` and `outputs/qwen2.5_3b_L24/generations/factuality_triviaqa.jsonl` where available.
- `all_mh_judged/`: pairwise judgments (winner and primary reason) for all six LLMs; raw generations are in
  `hf_results/*/` and `outputs/qwen2.5_3b_L24/generations/mental_health_800tok_diag.jsonl`.
- `hf_results/*/layer_selection/`, `outputs/qwen2.5_3b/layer_selection/`: layer-selection metrics.
  (`hf_results/gemma2_7b_L28` holds Gemma-2-27B at layer 28.)
- `outputs/figures_multimodel/`: summary CSVs, LaTeX tables, and figures for factuality, pairwise evaluation, and layer selection.
- `outputs/tts_ges_all_models/`: TTS and GES for all ten representations and k in {10, 15, 20, 30}.
- `outputs/tfidf_baseline/`: TF-IDF baseline metrics (30 splits).
- `outputs/<encoder>/`: per-encoder TTS, GES, containment, and trajectory statistics.
- `manifold_structure_study/results/`: PCA, TwoNN, MLE, reference distributions, and sampling-scale results.
- `dataset_agreement_test/`: blind LLM tier labels and pairwise energy judgments (by sentence/pair ID) and the resulting statistics.
