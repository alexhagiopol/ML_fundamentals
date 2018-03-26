function [J, grad] = costFunctionReg(theta, X, y, lambda)
	%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
	%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
	%   theta as the parameter for regularized logistic regression and the
	%   gradient of the cost w.r.t. to the parameters. 
	% ====================== YOUR CODE HERE ======================
	% Instructions: Compute the cost of a particular choice of theta.
	%               You should set J to the cost.
	%               Compute the partial derivatives and set grad to the partial
	%               derivatives of the cost w.r.t. each parameter in theta
	[m, n] = size(X);  % m is the number of examples, n is the number of parameters.
	h_theta_x = sigmoid(X * theta);
	J = sum(-1 .* y .* log(h_theta_x) - (1 - y) .* log(1 - h_theta_x)) / m + lambda * sum(theta(2:end) .^ 2) / (2*m);  % do not allow theta(1) to affect cost calculation
	grad = (((h_theta_x - y)' * X) + (lambda * [0; theta(2:end)])') / m;  % do not allow theta(1) to affect gradient calculation; regularization term transposed to be row vector & avoid column vector + row vector addition
	% =============================================================
end
