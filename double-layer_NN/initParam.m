function [W, b] = initParam(param)
% A function that gives initial values of parameters.
% ----------
% Arguments:
%   param: struct, parameters of the network
% Return:
%   W: cell, 1 X 2
%       W1: m X d (50 X 3072) 
%       W2: k X m (10 X 50)
%   b: cell, 1 X 2
%       b1: m X 1
%       b2: k X 1
    
    global NN2
    if nargin < 1     
        param = NN2;
    end

    W1 = 1/sqrt(param.d) * randn(param.m, param.d);
    b1 = zeros(param.m, 1);
    W2 = 1/sqrt(param.m) * randn(param.k, param.m);
    b2 = zeros(param.k, 1);
    
    W = {W1, W2};
    b = {b1, b2};
    
end
