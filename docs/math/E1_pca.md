> **Working notes: the checked numbers are in docs/report.md**

# E1  Principal component analysis (PCA) from scratch via the SVD
Code: `src/ext_E1_pca.py`. Results: `results/extE1_*.csv`; report section 8.1.

**Construction.** Standardise every feature (training mean and standard deviation) and centre it: Z (n x p). The SVD is Z = U S V'. The columns of V (p x p, orthogonal) are the **principal directions** (loadings): the first points along the direction of greatest variance in the data, the second along the greatest remaining variance orthogonal to it, and so on. The **scores** Z V = U S are the data in the new coordinates. The variance along component j is s_j^2/(n - 1) (the eigenvalues of the correlation matrix) and the **explained-variance ratio** is s_j^2 / sum_i s_i^2.

**Rank and collinearity.** The numerical rank is the number of singular values above a tolerance; a rank below p would mean exactly redundant features. The ratio of the largest to the smallest eigenvalue is the condition number of the correlation matrix. For two variables with correlation r the eigenvalues are 1 + r and 1 - r, so the first component captures a share (1 + |r|)/2 of their total variance: strongly correlated estradiol and progesterone would have a share near 1.

**Using it for prediction.** (i) *Ridge on all components.* V is orthogonal, so rotating the features changes neither the lengths of weight vectors nor the ridge penalty: ridge on all PCs gives exactly the same predictions as ridge on the standardised raw features (checked numerically). (ii) *Principal-component regression* keeps only the first k components and fits ordinary least squares; k is chosen by nested cross-validation. Truncation reduces variance but can discard low-variance directions that carry the outcome, because PCA ignores the outcome. (iii) *Trees on components*: trees are not rotation-invariant, so the result can change.

**No leakage.** The mean, standard deviation and V are computed from the training rows only, at every level (outer fold and every inner fold); held-out rows are merely projected with the training V. Fitting PCA on all rows first would let the held-out women influence the coordinate system.

**Caveat.** Components of a mix of dummy variables, hormones and questionnaire scores have no simple meaning; PCA is most informative here for the two hormones.
