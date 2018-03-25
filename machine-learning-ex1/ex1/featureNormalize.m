function [X_norm, mu, sigma] = featureNormalize(X)
	%FEATURENORMALIZE Normalizes the features in X 
	%   FEATURENORMALIZE(X) returns a normalized version of X where
	%   the mean value of each feature is 0 and the standard deviation
	%   is 1. This is often a good preprocessing step to do when
	%   working with learning algorithms.

	% Instructions: First, for each feature dimension, compute the mean
	%               of the feature and subtract it from the dataset,
	%               storing the mean value in mu. Next, compute the 
	%               standard deviation of each feature and divide
	%               each feature by its standard deviation, storing
	%               the standard deviation in sigma. 
	%
	%               Note that X is a matrix where each column is a 
	%               feature and each row is an example. You need 
	%               to perform the normalization separately for 
	%               each feature. 
	%
	% Hint: You might find the 'mean' and 'std' functions useful.
	%       

	% You need to set these values correctly

	mu = mean(X);
	sigma = std(X);
	input_size = size(X);
	X_norm = (X - ones(input_size)*diag(mu)) ./ (ones(input_size)*diag(sigma));  % see https://stackoverflow.com/a/5344435 for subtracting a vector from each row of matrix 
end
