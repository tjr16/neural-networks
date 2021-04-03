function W = Xavier(n_out, n_in)
% A function that initialize parameters 
% with Xavier distribution. 
% ----------
% Arguments:
%   n_out: output dimension of the layer, int
%   n_in: input dimension of the layer, int
% Return:
%   W: parameter matrix, n_in X n_out

    sigma = 1/sqrt(n_in);
    W = sigma * randn(n_out, n_in);
end
