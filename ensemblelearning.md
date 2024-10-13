Ensemble learning takes into account multiple classification algorithms into the classification problem. By using multiple models, you can take advantage of the different algorithms by themself, to make a better, more informed classification decision.

There are 3 types of ensemble learning:
- bagging
- boosting
- stacking

Boosting
----------
Boosting is where models are trained iteratively one after another, where future models are trained to fix the misclassifications that took place in the previous model. When a datapoint in the training/validation data is misclassified, the "importance weighting" of that point is increased, which means that future models will therefore "pay more attention" to that point to try and fix the error that was made.

The boosting algorithm stops after a number of iterations, or until a threshold is reached.

Pseudocode:
-Initialize weight vector with uniform weights
-Loop:
    -Apply weak leaner* to weighted training examples
    -Increase weight for misclassified examples
-Weighted majority voting on trained classifiers