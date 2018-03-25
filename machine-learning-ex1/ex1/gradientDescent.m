function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    %GRADIENTDESCENT Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    %   taking num_iters gradient steps with learning rate alpha

    % Initialize some useful values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);
    for iter = 1:num_iters
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %        
        h_theta = X * theta;
        theta = theta - alpha * (1 / m) * (X' * h_theta - X' * y);  % distribute multiplication of X_i. implement sum implicitly via mat mult
        % no need to implement different derivatives for theta_0 and theta_1 because X_0 is 1 so multiplying by it has no effect
        J_history(iter) = computeCost(X, y, theta);
    end
end
