function [W, b] = initParam(param, dist, sig)
% A function that initialize parameters.
% ----------
% Arguments:
%   param: [struct] parameters of the network
%   dist: [string] distribution (default: "He")
%   sig: standard deviation of a normal distribution
% Return:
%   W: cell, 1 X n_layers
%       W{i}: d_out X d_in
%   b: cell, 1 X n_layers
%       b{i}: d_out X 1
    
    global MLP    
    if nargin < 2
        dist = 'He';
    end
    if nargin < 1     
        param = MLP;
    end
    if nargin < 3
        assert(~strcmp(dist, 'Normal'));
    end

    
    switch dist
        case 'He'
            coef = sqrt(2./param.d);
        case 'Xavier'
            coef = sqrt(1./param.d);
        case 'Normal'
            coef = sig .* ones(size(param.d));
        otherwise
            coef = ones(size(param.d));
    end
             
    n_layers = numel(param.d) - 1;
    W = cell(1, n_layers);
    b = cell(1, n_layers);
    
    for i = 1: n_layers
        W{i} = coef(i) .* randn(param.d(i+1), param.d(i));
        b{i} = zeros(param.d(i+1), 1);
    end
    
end