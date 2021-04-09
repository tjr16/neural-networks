classdef DoubleLayer
% A value class as a double-layer neural network.

   %% === PROPERTIES ===
	properties
        H    % activation values, cell 1 X 2
        W    % weights, cell 1 X 2
        b    % bias, cell 1 X 2
        grad_W    % gradient of W, cell 1 X 2
        grad_b    % gradient of b, cell 1 X 2
    end
   
    %% === METHODS ===
    methods
      
        function obj = DoubleLayer(w, b)
            % Constructor of class DoubleLayer
            
            if nargin < 2
                [obj.W, obj.b] = initParam();
            else
                obj.W = w;
                obj.b = b;
            end

            obj.H = cell(1, 2);       
        end
      
        function obj = forward(obj, X)
            % Forward propagation, store activation values.
            % ----------
            % Arguments:
            %   X: image pixel data, d X n (3072 X 10000) 
            %      double, [0, 1]

            s1 = obj.W{1} * X + obj.b{1};  % linear1
            s1(s1<0) = 0;  % activation1: ReLu
            obj.H{1} = s1;
            s2 = obj.W{2} * obj.H{1} + obj.b{2};   % linear2
            obj.H{2} = softmax(s2);    % activation2: Softmax
        end

        function obj = computeGradients(obj, X, Y, lambda)
            % A function that evaluates, for a mini-batch,
            % the gradients of the cost function wrt W and b.
            % BOTH forward and backward propagation are done.
            % ----------
            % Arguments:
            %   X: image pixel data, d X nb (3072 X batch_size) 
            %      double, [0, 1]
            %   Y: k X nb (10 X batch_size), one-hot
            %   lambda: L2 penalty coefficient, hyperparamter
            % Return:
            %   grad_W: cell, 1 X 2
            %   grad_b: cell, 1 X 2
            
            nb = size(X, 2);	% batch size
            
            % forward computation
            obj = obj.forward(X);
 
            % second layer
            G = obj.H{2} - Y;	% dL/dz2, z2 = W2 * X2 + b2
            grad_W2 = G * obj.H{1}'/ nb + 2 * lambda * obj.W{2};    % dL/dW2
            grad_b2 = G * ones(nb, 1) / nb; % dL/db2
            
            % first layer
            G = obj.W{2}' * G;  % dL/dX2, X2 = H1
            G = G .* (obj.H{1} > 0);    % dL/dz1, z1 = W1 * X1 + b1
            grad_W1 = G * X'/ nb + 2 * lambda * obj.W{1};   % dL/dW1
            grad_b1 = G * ones(nb, 1) / nb;	% dL/db1
            
            obj.grad_W = {grad_W1, grad_W2};
            obj.grad_b = {grad_b1, grad_b2};
        end
        
        function output = output(obj)
            % A function that returns output of the network
            
            output = obj.H{2};
            if isempty(output)
                error('Empty output!');
            end
        end
        
        function penalty = penaltyL2(obj, lambda)
            % L2 regularization term in cost function

            penalty = 0;
            for i = 1: numel(obj.W)
                penalty = penalty + lambda * sum(obj.W{i} .^ 2, 'all');
            end
        end
        
    end
end