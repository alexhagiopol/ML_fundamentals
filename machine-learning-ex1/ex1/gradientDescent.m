function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
    for iter = 1:num_iters
        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %
        X_concat_y = horzcat(X,y);  % create mx3 matrix to multiply with theta 3x1 matrix
        theta_vec = [theta; -1];  % Append -1 to yield -y after mat mult: [theta_0; theta_1; -1]
        h_theta_minus_y = X_concat_y * theta_vec;  % mx1 vector representing values of h_theta(x_i) - y_i for each example i
        d_d_theta_0 = sum(h_theta_minus_y) * (1/m);  % derivative of cost w.r.t. theta_0
        d_d_theta_1 = sum(h_theta_minus_y .* X(:,2)) * (1/m);  % derivative of cost w.r.t. theta_1
        theta(1) = theta(1) - alpha * d_d_theta_0;  % update step for theta_0
        theta(2) = theta(2) - alpha * d_d_theta_1;  % update step for theta)_1
        % ============================================================
        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);
    end
end
