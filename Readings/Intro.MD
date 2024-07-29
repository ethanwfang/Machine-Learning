## What is ML?
Machine Learning is about making machines get better at some task by learning from data, instead of having to explicitly code rules. There are many types of machine learning systems: unsupervised vs. supervised, batch vs. online, instance-based or model-based.

The typical route a machine learning engineer takes is
- gather data in a training set, and feed the training set to a learning algorithm
- if the model is model-based, it tunes to some parameters to fit the model to the training set to hopefully make good predictions on new cases as well
- if the algorithm is instanced-based, it just learns the examples by ehart and generalizes to new instances by using a smimilarity measure to compare them to the learned instances

The system will not perform well if your training set is too small, or if the data is not representative, is noisy, or is polluted with irrelevant features. This stresses the importance of the quantity, quality, and manipulation of training data. In addition, the model that you choose to train on the data cannot be too simple, and cannot be too complex.

After training a model, you also have to evaluate it and fine-tune it if necessary.
