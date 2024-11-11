# Ensemble Learning
Ensemble learning takes into account multiple classification algorithms into the classification problem. By using multiple models, you can take advantage of the different algorithms by themself, to make a better, more informed classification decision.

There are 3 types of ensemble learning:
- bagging
- boosting
- stacking

### Boosting
Boosting is where models are trained iteratively one after another, where future models are trained to fix the misclassifications that took place in the previous model. When a datapoint in the training/validation data is misclassified, the "importance weighting" of that point is increased, which means that future models will therefore "pay more attention" to that point to try and fix the error that was made.

The boosting algorithm stops after a number of iterations, or until a threshold is reached.

Pseudocode:    
-Initialize weight vector with uniform weights    
-Loop:     
&nbsp;&nbsp;&nbsp;&nbsp;-Apply weak leaner* to weighted training examples    
&nbsp;&nbsp;&nbsp;&nbsp;-Increase weight for misclassified examples    
-Weighted majority voting on trained classifiers    

### Bagging: bootstrap aggregating
Bagging is where you train multiple classifiers of the same type on the same data. In this way, you need to force a way for the classifiers to be different, you use something called a Bootstrap sample. This is the concept where you randomly sample with replacement, from a given dataset. This way, some examples are left out and some are repeated multiple times.

Why does bagging work?

Each bootstrap sample is a new training data set drawn idependently and identically distributed from the original training data. Since each sample is different from the others, that means that there is a different decision boundary, and should have much less variance than the original classifier.

### Random Forest Classifier
A random forest is a collection of bagged decision trees with a random subset of features considered at each split.

This forces trees to look even more different from each other, and also reduces computational time for training (less splits to consider). 

1. Draw a random bootstrap sample of size n (randomly choose n samples from the training dataset with replacement). 
2. Grow a decision tree from the bootstrap sample. Each each node:     
(a) Randomly select features without replacement.    
(b) Split the node using the feature that provides that best split according to the objective function (e.g. maximizing the information gain)    
3. Repeat steps 1-2 "k" times 
4. Aggregate prediction by each tree to assign the class label by majority vote.

