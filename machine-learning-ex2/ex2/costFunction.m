function [J, grad] = costFunction(theta, X, y)
	%COSTFUNCTION Compute cost and gradient for logistic regression
	%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
	%   parameter for logistic regression and the gradient of the cost
	%   w.r.t. to the parameters.
	% ====================== YOUR CODE HERE ======================
	% Instructions: Compute the cost of a particular choice of theta.
	%               You should set J to the cost.
	%               Compute the partial derivatives and set grad to the partial
	%               derivatives of the cost w.r.t. each parameter in theta
	%
	% Note: grad should have the same dimensions as theta
	% Note: X is an mxn matrix where n = 3 in the given dataset: 2 multiplicative params and one intercept
	% Note: y is an mx1 vector
	%
	[m, n] = size(X);  % m is the number of examples, n is the number of parameters.
	J = sum(-1 .* y .* log(sigmoid(theta' .* ))) / m



	grad = zeros(size(theta));
	% =============================================================
end
