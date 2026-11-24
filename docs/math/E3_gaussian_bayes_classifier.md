> **Working notes: the checked numbers are in docs/report.md**

# E3  Gaussian Bayes classifier for the cycle phase
Code: `src/ext_E3_gaussian_bayes.py`. Results: `results/extE3_*.csv`; report section 8.3.

**Bayes' theorem for classification.** For classes k (menses, peri-ovulatory, luteal) and features x = (log estradiol, log progesterone):
p(k | x) = pi_k p(x | k) / sum_j pi_j p(x | j).
pi_k is the **prior** (how common class k is), p(x | k) is the **likelihood** (how typical x is for class k), the denominator makes the probabilities add to 1. The predicted class is the one with the largest posterior.

**Gaussian likelihoods.** Model each class as a bell-shaped cloud: p(x | k) = N(x; mu_k, Sigma_k), where mu_k is the class mean and Sigma_k its covariance, estimated by maximum likelihood (class means and covariances divided by the class size, as scikit-learn does) in the **training fold only**. The log of the numerator is
log pi_k - (1/2)(x - mu_k)' Sigma_k^-1 (x - mu_k) - (1/2) log|Sigma_k| + constant.
Posteriors are computed in log space and normalised with the log-sum-exp trick so that nothing underflows.

**Three covariance choices.** *Full* (one covariance per class; quadratic decision boundaries; = QDA). *Shared* (one pooled covariance; the quadratic terms cancel and the boundaries are straight lines; = LDA). *Diagonal* (no correlation between the two hormones inside a class; naive Bayes). More flexible = more parameters to estimate from the training women.

**Priors and balanced accuracy.** With *empirical priors* (class frequencies in the training fold) the classifier leans towards common classes, which maximises plain accuracy. Balanced accuracy averages the recall of the classes, which corresponds to *equal priors*. Both versions are reported; the primary analysis uses the empirical priors, as specified.

**Relation to logistic regression.** LDA's posterior has exactly the form of a logistic (softmax) regression; QDA adds quadratic terms. Logistic regression fits the boundary directly without modelling x; the Gaussian Bayes classifier models how x is distributed within each class and can be better when the Gaussian assumption holds and the sample is small.

**Checks.** The full-covariance classifier is compared numerically with scikit-learn's QDA (same posteriors to machine precision). Priors, means and covariances never see the held-out woman.
