function [W, b] = initParam(param)
% A function that initialize parameters using He initialization.
% ----------
% Arguments:
%   param: struct, parameters of the network
% Return:
%   W: cell, 1 X n_layers
%       W{i}: d_out X d_in
%   b: cell, 1 X n_layers
%       b{i}: d_out X 1
    
    global MLP
    if nargin < 1     
        param = MLP;
    end
    
    n_layers = numel(param.d) - 1;
    W = cell(1, n_layers);
    b = cell(1, n_layers);
    
    for i = 1: n_layers
        W{i} = sqrt(2/param.d(i)) * randn(param.d(i+1), param.d(i));
        b{i} = zeros(param.d(i+1), 1);
    end
    
end