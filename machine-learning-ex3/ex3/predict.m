function p = predict(Theta1, Theta2, X)
	%PREDICT Predict the label of an input given a trained neural network
	%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	%   trained weights of a neural network (Theta1, Theta2)

	% ====================== YOUR CODE HERE ======================
	% Instructions: Complete the following code to make predictions using
	%               your learned neural network. You should set p to a 
	%               vector containing labels between 1 to num_labels.
	%
	% Hint: The max function might come in useful. In particular, the max
	%       function can also return the index of the max element, for more
	%       information see 'help max'. If your examples are in rows, then, you
	%       can use max(A, [], 2) to obtain the max for each row.
	%
	% Useful values
	m = size(X, 1);  % number of examples
	n = size(X, 2);  % number of features not including 1s
	X = [ones(m, 1), X];  % append ones to X representing bias terms
	z2 = Theta1 * X';
	a2 = sigmoid(z2)';
	a2 = [ones(size(a2, 1),1) a2];  % append ones to a2 representing bias terms
	z3 = Theta2 * a2';
	a3 = sigmoid(z3);
	[max_values, p] = max(a3);  % p will be the index of the feature with the highest sigmoid activation value
	% =========================================================================
end
