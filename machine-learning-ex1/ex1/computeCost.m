function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

m = length(y); % number of training examples
h_theta = horzcat(X,y);  % create mx3 matrix to multiply with theta 3x1 matrix
theta = [theta; -1];  % Append -1 to yield -y after mat mult: [theta_0; theta_1; -1]
J = sum((h_theta * theta).^2) / (2 * m);  % h_theta * theta = h_theta(x_i) - y_i for each i
% =========================================================================
end
