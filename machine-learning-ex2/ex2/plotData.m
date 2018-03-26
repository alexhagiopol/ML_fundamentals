function plotData(X, y)
	%	PLOTDATA Plots the data points X and y into a new figure 
	%   PLOTDATA(x,y) plots the data points with + for the positive examples
	%   and o for the negative examples. X is assumed to be a Mx2 matrix.
	% Create New Figure
	% ====================== YOUR CODE HERE ======================
	% Instructions: Plot the positive and negative examples on a
	%               2D plot, using the option 'k+' for the positive
	%               examples and 'ko' for the negative examples.
	%
	figure; 
	hold on;
	positiveDataIndices = find(y == 1);
	negativeDataIndices = find(y == 0);
	plot(X(positiveDataIndices, 1), X(positiveDataIndices, 2), 'b+');  % ignore instructions and use blue + for acceptances
	plot(X(negativeDataIndices, 1), X(negativeDataIndices, 2), 'ro');  % ignore instructions and use red circle for acceptances
	hold off;
	% =========================================================================
end
