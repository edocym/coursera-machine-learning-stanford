function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

C_vals     = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vals = [0.01 0.03 0.1 0.3 1 3 10 30];
error_min = inf;


for i = 1:length(C_vals)
  for j = 1:length(sigma_vals)
    model = svmTrain(X, y, C_vals(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vals(j)));
    predictions = svmPredict(model, Xval);
    prediction_error(i, j) = mean(double(predictions ~= yval));
    
    if (prediction_error(i, j) <= error_min)
      C_opt = C_vals(i);
      sigma_opt = sigma_vals(j);
      error_min = prediction_error(i, j);
    endif
  
  endfor
endfor
#{
[colmin, rowindex] = min(prediction_error);
[minval, colindex] = min(colmin);

C = C_vals(rowindex(colindex));
sigma = sigma_vals(colindex);
#}
C = C_opt;
sigma = sigma_opt;


end
