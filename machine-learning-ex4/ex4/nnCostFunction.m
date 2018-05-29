function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
	%NNCOSTFUNCTION Implements the neural network cost function for a two layers
	%neural network which performs classification
	%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
	%   X, y, lambda) computes the cost and gradient of the neural network. The
	%   parameters for the neural network are "unrolled" into the vector
	%   nn_params and need to be converted back into the weight matrices. 
	% 
	%   The returned parameter grad should be a "unrolled" vector of the
	%   partial derivatives of the neural network.
	%
	% ====================== YOUR CODE HERE ======================
	% Instructions: You should complete the code by working through the
	%               following parts.
	%
	% Part 1: Feedforward the neural network and return the cost in the
	%         variable J. After implementing Part 1, you can verify that your
	%         cost function computation is correct by verifying the cost
	%         computed in ex4.m
	%
	% Part 2: Implement the backpropagation algorithm to compute the gradients
	%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
	%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
	%         Theta2_grad, respectively. After implementing Part 2, you can check
	%         that your implementation is correct by running checkNNGradients
	%
	%         Note: The vector y passed into the function is a vector of labels
	%               containing values from 1..K. You need to map this vector into a 
	%               binary vector of 1's and 0's to be used with the neural network
	%               cost function.
	%
	%         Hint: We recommend implementing backpropagation using a for-loop
	%               over the training examples if you are implementing it for the 
	%               first time.
	%
	% Part 3: Implement regularization with the cost function and gradients.
	%
	%         Hint: You can implement this around the code for
	%               backpropagation. That is, you can compute the gradients for
	%               the regularization separately and then add them to Theta1_grad
	%               and Theta2_grad from Part 2.
	%
	% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	% for our 2 layer neural network
	
	% *** FORWARD PROPAGATION ***
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));
	% Setup some useful variables
	m = size(X, 1);
	% make one-hot encoded y vector
	y_one_hot = zeros(m, num_labels);
	I = eye(num_labels);
	for i = 1:m
		y_one_hot(i,:) = I(y(i), :);
	end
	% compute prediction h_theta_x
	a1 = [ones(m,1), X]; % add column of ones representing bias unit to X matrix representing
	z2 = a1*Theta1';
	a2 = sigmoid(z2);
	a2 = [ones(m,1), a2];  % add bias unit
	a3 = h_theta_x = sigmoid(a2 * Theta2');
	% compute unregularized cost
	J = (1/m)*sum(sum(-y_one_hot.*log(h_theta_x) - (1-y_one_hot).*log(1-h_theta_x)));
	% compute regularized cost
	J += (lambda / (2*m))*(sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

	% *** BACK PROPAGATION ***
	delta3 = a3 - y_one_hot;  % part 2 of instructions
	delta2 = (( delta3 * Theta2 ) .* sigmoidGradient([ones(m, 1), z2]))(:, 2:end);  % part 3 of instructions
	Theta1_grad = (delta2' * a1) ./ m;  % parts 4 and 5 of instructions. same as DELTA1
	Theta2_grad = (delta3' * a2) ./ m;  % parts 4 and 5 of instructions. same as DELTA2
	Theta1_grad += (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)]  % implement regularization step in section 2.5
	Theta2_grad += (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)]  % implement regularization step in section 2.5
	% Unroll gradients
	grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
