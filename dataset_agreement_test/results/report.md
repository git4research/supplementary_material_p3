# Dataset agreement test -- results

Dataset sentences: 636; sentences quoted in a rubric: 0 []

## Part 1 (tier) -- gpt5
- completeness: {'parsed': 636, 'missing': 0, 'conflicting': 0, 'repeated_same': 0, 'reask_file': None}
- n=636: exact 80.2%, within one 94.7%, kappa 0.768, quadratic kappa 0.951, Spearman (tier) 0.955, Spearman (judge tier vs score) 0.944
- 95% CIs: {'exact': [0.7720125786163522, 0.8302279874213835], 'within_one': [0.9292059748427672, 0.9638364779874213], 'kappa_quadratic': [0.9381600808821252, 0.9619882178929939], 'spearman_tier': [0.9425785290382861, 0.9659787998406638]}
- excluding rubric examples (n=636): exact 80.2%, within one 94.7%, quadratic kappa 0.951
- recall by tier: Collapse 85.3, Striving 77.9, Conflict 90.7, Activation 59.8, Growth 65.3, Clarity 88.5, Unity 93.1
- top confusions: Activation/Growth (38), Striving/Conflict (18), Growth/Clarity (16), Collapse/Striving (16), Activation/Clarity (14)

## Part 1 (tier) -- opus5
- completeness: {'parsed': 636, 'missing': 0, 'conflicting': 0, 'repeated_same': 0, 'reask_file': None}
- n=636: exact 81.9%, within one 95.9%, kappa 0.788, quadratic kappa 0.961, Spearman (tier) 0.963, Spearman (judge tier vs score) 0.953
- 95% CIs: {'exact': [0.789308176100629, 0.8474842767295597], 'within_one': [0.9433962264150944, 0.9732704402515723], 'kappa_quadratic': [0.9508718725712954, 0.9702619853525216], 'spearman_tier': [0.9530557685521432, 0.9713307639884254]}
- excluding rubric examples (n=636): exact 81.9%, within one 95.9%, quadratic kappa 0.961
- recall by tier: Collapse 85.3, Striving 79.8, Conflict 90.7, Activation 68.2, Growth 69.4, Clarity 86.9, Unity 91.6
- top confusions: Activation/Growth (40), Collapse/Striving (17), Striving/Conflict (16), Growth/Clarity (15), Growth/Unity (10)

## Part 2 (energy) -- gpt5
- completeness: {'parsed': 1000, 'missing': 0, 'conflicting': 0, 'repeated_same': 0, 'reask_file': None}
- concordance 92.2% (920/998 non-tied pairs; 2 score-tied excluded), 95% CI ['90.5', '93.8'], binomial p = 2.8e-183
- cross-tier pairs: 96.9% (n=862), 95% CI ['95.7', '97.9']; same-tier pairs: 62.5% (n=136), 95% CI ['54.4', '70.6'], binomial p = 0.00448
- by gap: <1: 63.9% (n=183), 1-2: 91.9% (n=135), 2-4: 99.7% (n=303), 4-8: 100.0% (n=338), >=8: 100.0% (n=39)
- excluding pairs with a rubric example: 92.2% (n=998)
- answered 'A': 49.3% (true rate of A higher is 48.9%)

## Part 2 (energy) -- opus5
- completeness: {'parsed': 1000, 'missing': 0, 'conflicting': 0, 'repeated_same': 0, 'reask_file': None}
- concordance 92.5% (923/998 non-tied pairs; 2 score-tied excluded), 95% CI ['90.9', '94.0'], binomial p = 1.63e-186
- cross-tier pairs: 97.2% (n=862), 95% CI ['96.1', '98.3']; same-tier pairs: 62.5% (n=136), 95% CI ['54.4', '70.6'], binomial p = 0.00448
- by gap: <1: 65.6% (n=183), 1-2: 91.1% (n=135), 2-4: 100.0% (n=303), 4-8: 100.0% (n=338), >=8: 100.0% (n=39)
- excluding pairs with a rubric example: 92.5% (n=998)
- answered 'A': 49.7% (true rate of A higher is 48.9%)

## Consensus disagreements (all judges agree with each other, not with the original tier)
- n = 88 (listed in results/consensus_disagreements.csv; worth a manual review)
- by original -> judged tier: Activation->Growth (18), Growth->Activation (10), Striving->Collapse (10), Growth->Clarity (7), Striving->Conflict (7), Activation->Clarity (6), Collapse->Striving (6), Unity->Growth (6), Clarity->Growth (5), Collapse->Conflict (5), Conflict->Striving (5), Activation->Unity (2), Clarity->Unity (1)

## Between judges
{'tier gpt5 vs opus5': {'n': 636, 'exact': 0.9150943396226415, 'kappa_quadratic': 0.9756732811716107}, 'energy gpt5 vs opus5': {'n': 1000, 'same_choice': 0.966}}
