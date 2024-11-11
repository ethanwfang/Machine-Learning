# Evaluation Metrics and Voting Classifiers

### Confusion Matrix
|  Labels | P | N |
|-----------|----------|----------|
|  P   | True Positive  | False Negative |
| N  | False Positive | True Negative  |

True Positive Rate (TPR): $\frac{TP}{TP + FN}$

False Positive Rate (FPR): $\frac{FP}{FP+TN}$

Precision (PRE) : $\frac{TP}{TP+FP}$

Precision & Recall combine into the F_{1} score = $2\frac{PRE*REC}{PRE+REC}$

### Reciever Operating Characteristic (ROC) curves
An ROC curve is a graphical representation used to evaluate the performance of a binary classification model. It shows the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR) at various classification thresholds.

Interpretation:
- The closer the ROC curve is to the top-left corner, the better the model.
- A random classifier (flipping a coin or something) produces a diagonal line from (0,0) to (1,1)
- The Area Under the ROC Curve (AUC) is a single scalar value summarizing the overall performance. AUC ranges from 0 to 1, where 1 means perfect, 0.5 is random, and less than 0.5 is worse than random. 

### Multiclass classification metrics
Multiclassification metrics focus on how performance metrics are calculated when dealing with more than two classes.

One such example is:
1. One-vs-all (OvA) Evaluation
In multiclass classification, metrics like precision and recall are often calculated using a one-vs-all approach. For each class, th emodel treats it as the positive class and all other classes as the negative class, effectively reducing the problem to a series of binary classifications.

2. Micro-Averaged Precision
This aggregates contributions from all classes by summing up the individual true positives (TP), false positives (FP), and false negatives (FN) across classes

$PRE_{micro} = \frac{TP_{1} + TP_{2} + ... + TP_{k}}{TP_{1} + FP_{1} + TP_{2} + FP_{2} + ... + TP_{k} + FP_{k}}$

Characteristics of Micro PA:
- treats every example equally, irrespective of the class
- tends to favor classes with a large number of examples because it aggregates all instances

3. Macro-Averaged Precision
This calculates the precision for each class independently and then takes the average across all classes.

$PRE_{macro} = \frac{PRE_1 + PRE_2 + ... + PRE_k}{k}$, where $PRE_i = \frac{TP_i}{TP_i + FP_i}$

Characteristics of Macro PA:
- treats every class equally, regardless of how many examples belong to it
- less sensitive to class imbalance since every class contributes equally to the average

#### Micro vs. Macro?
Micro puts more weight on larger classes since it aggregates over all instances, while Macro gives equal importance to all classes regardless of their size, which can be more informative for imbalanced datasets. 

### How do you deal with class imbalance?
You need to use an evaluation metric that accounts for class imbalance (The ROC curve is usually a good choice for this).

Class imbalance also affects classifier training, since minimizing total loss biases classifier towards the majority class. 

There are a variety of ways to counteract this, such as:
- assigning a larger penalty to wrong predictions on the minority class
- upsampel minority class
- downsample majority class
- generate synthetic training examples for minority class 







