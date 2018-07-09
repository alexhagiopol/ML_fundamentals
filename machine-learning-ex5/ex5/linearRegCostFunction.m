function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
	% LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
	% regression with multiple variables
	%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
	%   cost of using theta as the parameter for linear regression to fit the 
	%   data points in X and y. Returns the cost in J and the gradient in grad
	% ====================== YOUR CODE HERE ======================
	% Instructions: Compute the cost and gradient of regularized linear 
	%               regression for a particular choice of theta.
	%
	%               You should set J to the cost and grad to the gradient.
	%
	
	m = length(y); % number of training examples
	h_theta_x = X * theta;
	J = (1 / (2 * m)) * (sum((h_theta_x - y) .^ 2) + lambda * sum(theta(2:end) .^ 2));  % regularized linear regression cost function
	% gradient = multivar derivative of the cost w.r.t. theta 
	grad = zeros(size(theta));
	% X(:,1) is equal to 1 because theta_0 is a constant bias term; no need to multiply by X(:,1)
	grad(1) = (1 / m) * sum(h_theta_x - y)';  % gradient w.r.t. theta_0 is unregularized
	% sum is implicit in matrix multiplication by X; no need for sum operator
	grad(2:end) = (1 / m) * ((h_theta_x - y)' * X(:, 2:end) + lambda * theta(2:end)');
	% =========================================================================
end 
