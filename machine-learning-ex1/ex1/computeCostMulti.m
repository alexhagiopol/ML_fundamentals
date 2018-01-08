function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% N.B.
% 1. X is an mx(n+1) matrix whose first n columns contain the actual input data values and
% whose last column contains all 1s to be multiplied with the theta_0 intercept term.
% 2. Y is an mx1 matrix containing output data points
% 3. theta is a nx1 matrix containing the current parameter estimate

m = length(y); % number of training examples
X_concat_y = horzcat(X, y);
theta_vec = [theta; -1]; % append -1 to yield -y after mat mult: [theta_0; ... theta_n; -1]
J = sum((X_concat_y * theta_vec).^2) / (2 * m);


% =========================================================================
end
