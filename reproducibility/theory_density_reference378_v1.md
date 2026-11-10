# Deterministic all-slot reference

Let D be a declared relation domain and P its listed-positive reference, with P contained in D. The rule reports every element of D and uses the domain alone.

Its counts are TP=|P|, FP=|D|-|P| and FN=0. Its precision is |P|/|D| and its F1 is 2|P|/(|D|+|P|). These identities follow directly from the score definitions in Appendix D.2.

For the optical catalogue, thirteen domains each contain nine slots. Twelve references contain all nine slots, and one reference is empty. The pooled counts are TP=108, FP=9 and FN=0. Consequently micro F1=24/25 and macro F1=12/13.

This analytical reference measures the effect of the catalogue's reference density. Comparing its precision and recall with the measured estimator explains the consequences of its false positives and false negatives. The computation uses the frozen domains and reference labels, together with the delivered estimator's counts for its direct comparison.
