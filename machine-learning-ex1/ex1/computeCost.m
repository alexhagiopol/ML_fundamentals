function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% N.B.
% 1. X is an mx2 matrix whose first column contains the actual input data values and
% whose second column contains all 1s to be multiplied with the theta_0 intercept term.
% 2. Y is an mx1 matrix containing output data points
% 3. theta is a 2x1 matrix containing the current parameter estimate [theta_0; theta_1]

m = length(y); % number of training examples
X_concat_y = horzcat(X,y);  % create mx3 matrix to multiply with theta 3x1 matrix
theta_vec = [theta; -1];  % Append -1 to yield -y after mat mult: [theta_0; theta_1; -1]
J = sum((X_concat_y * theta_vec).^2) / (2 * m);  % h_theta * theta = h_theta(x_i) - y_i for each i
% =========================================================================
end
