# Week 6: Advice for Applying Machine Learning

## Part 1: Deciding What to Try Next

2. If a machine learning model yields unacceptable results, what are the next
steps? Most people just try random things based on intuition (6+ months).

		Get more training examples.
		Try smaller sets of features.
		Try getting additional features.
		Try adding polynomial features.
		Try decreasing LAMBDA.
		Try incresing LAMBDA.

3. How do you choose what to do in a systematic way? Conclusion: you need a numerical dignostic to inform this decision.

## Part 2: Evaluating a Learned Hypothesis with a Diagnostic

5. In slide 5, we have a model that is very complex and fails to generalize to data outside the training set. In this plot, the overfitting effect is easy to visualize. But in an N-dimensional space (like in a neural network) overfitting is very difficult to visualize. 

6. Thus we divide our dataset into training and test sections. We learn our hypothesis parameters THETA by first minimizing error on the training set. We do not train on the test set and use it only as an evaluation of performance.

7. Definition of test set error.

## Part 3: Model Selection

10. Once the parameters THETA are fit to the training set, it's very likely that the test set error ("generalization error") is greater than the training set error.

11. Systematic model selection. How would we choose the degree of polynomial to use for our model? Train several different models on the training set, evaluate each one on the test set, then choose the one with the smallest test set error.
`The problem with this is that we are effectively fitting the parameter d 
(degree) to the test set.` 

12. This is why people instead divide their datasets into training (20%), validation (20%), and test (20%).

13. Definitions of training, validation, and test error.

14. Thus we define a new model selection procedure. Train several different models on the training set. Pick the one that gets the lowest `validation error`, then report its error on the test set. Thus the test set error remains representative of generalization error.

## Part 4: Diagnosing Bias vs Variance

16. 
		bias = underfitting = both training and validation errors are `HIGH`. 

		variance = overfitting = training error is `LOW` while validation error is `HIGH`.

17. As we increase the degree of the polynomial, training and validation error start out both high because the model does not have the capacity to represent the phenomenon in the data. As the degree of the polynomial increases, training error decreases monotonically because the model eventually just memorizes the dataset. Validation error peaks at a minimum value that is "just right", but then goes back up once overfitting begins.

18. 
		bias: training error is high. training error approx = validation error.
		
		variance: training error is low. validation error is high.

## Part 5: Regularization, Bias, and Variance

20. How does LAMBDA (regularization) impact bias and variance? 
		
		Large LAMBDA        = high bias (underfit)
		Intermediate LAMBDA = just right
		Small LAMBDA        = high variance (overfit)

21. LAMBDA can be chosen similarly to the degree of the polynomial discussed above. Compute validation error after training models with multiple values of LAMBDA.

## Part 6: Learning Curves

25. Plot training and validation error as a function of the number of training set size (m) to produce a learning curve. Notice how when m is small, training error is very low because it's easy for any model to fit a small amount of data. 

26. `High bias learning curve.` Training and validation errors converge to high values. Training error remains slightly lower than validation error. Adding more training examples does not improve training or validation error.

27. `High variance learning curve.` Validation error is greater than training error. Gap between training and validation error becomes smaller as trainign examples are added. 

## Part 7: Deciding What to do Next (Revisited)

29. 
		Get more training examples.      -> Fixes high variance.
		Try smaller sets of features.    -> Fixes high variance.  
		Try getting additional features. -> Fixes high bias.
		Try adding polynomial features.  -> Fixes high bias.
		Try decreasing LAMBDA.           -> Fixes high bias.
		Try incresing LAMBDA.            -> Fixes high variance.

30. Small neural networks are computationally cheap and prone to underfitting. Large neural networks are computationally expensive and prone to overfitting. Use regularization and massive datsets to fix overfitting.
  