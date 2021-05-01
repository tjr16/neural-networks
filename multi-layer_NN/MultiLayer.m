classdef MultiLayer
% A value class as a multi-layer neural network.

    %% === PROPERTIES ===
    properties
        n_layers  % #layers
        % param
        W         % weights, cell 1 X n_layers
        b         % bias, cell 1 X n_layers
        % forward
        X         % input values of each layer, cell 1 X n_layers
        X_out     % output activate value
        % backward
        grad_W    % gradient of W, cell 1 X n_layers
        grad_b    % gradient of b, cell 1 X n_layers
        % evaluation mode
        eval_mode
        % dropout
        dropout_mode
        prob      % probability of an element to be zeroed
        % batch_normalization
        bn_mode    % boolean
        bn_layers  % [cell]: BatchNormLayer 
    end
   
    %% === METHODS ===
    methods
      
        function obj = MultiLayer(w, b, dropout, bn)
            % Constructor of class DoubleLayer
            % ----------
            % Arguments:
            %   w, b: network parameter matrices
            %   dropout: [double] probability of an element to be zeroed
            %       range: 0 < dropout < 1
            %       input "[]" to ignore this parameter
            %   bn: [logical] use batch normalization
            %       input "[]" to ignore this parameter
            
            global MLP
            obj.n_layers = numel(MLP.d) - 1;
            
            % analyze parameters
            % FIXME: add var and mean
            % bn
            if nargin < 4  % no param
                obj.bn_mode = false;
            elseif isempty(bn)  % empty param: []
                obj.bn_mode = false;
            elseif ~islogical(bn)
                error('Invalid parameter: bn');
            else
                obj.bn_mode = bn;
                obj.bn_layers = cell(obj.n_layers-1, 1);
                for i = 1: numel(obj.bn_layers)
                    obj.bn_layers{i} = BatchNormLayer(MLP.d(i+1));
                end
            end
            
            % dropout
            if nargin < 3  % no param
                obj.prob = 0;
                obj.dropout_mode = false;
            elseif isempty(dropout)  % empty param: []
                obj.prob = 0;
                obj.dropout_mode = false;
            elseif dropout <= 0 || dropout >= 1
                error('Invalid parameter: dropout. Range: (0, 1)');
            else
                obj.prob = dropout;
                obj.dropout_mode = true;
            end
            
            if nargin < 2
                [obj.W, obj.b] = initParam();
            else
                obj.W = w;
                obj.b = b;
            end

            obj.X = cell(1, obj.n_layers);
            obj.eval_mode = false;
        end
      
        function [obj, mask] = forward(obj, X)
            % Forward propagation, store activation values.
            % ----------
            % Arguments:
            %   X: image pixel data, d X n (3072 X 10000) 
            %      double, [0, 1]
            % Return:
            %   obj: network with output
            %   mask: dropout mask, element: 0 or 1/(1-p)
            
            obj.X{1} = X;   % input
            mask = 0;  
            
            % hidden layers
            for i = 1: obj.n_layers - 1
                
                % 1. linear transform
                s = obj.W{i} * obj.X{i} + obj.b{i};  % linear
                
                % 2. batch normalization
                if obj.bn_mode
                    obj.bn_layers{i} = ...
                        obj.bn_layers{i}.forward(s, obj.eval_mode);
                    s = obj.bn_layers{i}.s_out;
                end
                
                % 3. activation
                s(s < 0) = 0;    % ReLU
                obj.X{i+1} = s;  % to next layer, last one: X{n_layer}
                
                % 4. dropout
                if ~obj.eval_mode && obj.dropout_mode
                    % training mode
                    % dropout: reserve 1-p, scale
                    p = obj.prob;
                    mask = (rand(size(obj.X{i+1})) > p)/(1-p);
                    obj.X{i+1} = obj.X{i+1} .* mask;
                else
                    mask = ones(size(obj.X{i+1}));
                end
            end

            % output layer
            s = obj.W{end} * obj.X{end} + obj.b{end};  % linear
            obj.X_out = softmax(s);  % Softmax
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
            
            nb = size(X, 2);	% batch size
            
            % forward
            obj.eval_mode = false;
            [obj, mask] = obj.forward(X);
            
            % backward
            obj.grad_W = cell(1, obj.n_layers);
            obj.grad_b = cell(1, obj.n_layers);
            
            % output layer
            G = obj.X_out - Y;   % dL/dz, z = W * X + b
            obj.grad_W{end} = G * obj.X{end}'/ nb + ...
                2 * lambda * obj.W{end};    % dL/dW
            obj.grad_b{end} = G * ones(nb, 1)/ nb;

            % hidden layers
            for i = obj.n_layers: -1: 2
                G = obj.W{i}' * G;  % dL/dX(i)
                
                % 4. dropout and 3. activation
                if obj.dropout_mode
                    G = G .* (obj.X{i} > 0) .* mask;  % dL/dZ(i-1)
                else
                    G = G .* (obj.X{i} > 0);
                end
                
                % 2. batch normalization
                if obj.bn_mode
                    % NOTE: i-1 (no BN in output layer)
                    obj.bn_layers{i-1} = obj.bn_layers{i-1}.computeGradient(G);
                    G = obj.bn_layers{i-1}.grad_s;
                end
                
                % 1. linear
                obj.grad_W{i-1} = G * obj.X{i-1}'/ nb + ...
                    2 * lambda * obj.W{i-1};   % dL/dW(i-1)
                obj.grad_b{i-1} = G * ones(nb, 1) / nb;  % dL/db(i-1)   
            end
        end
        
        function out = output(obj)
            % A function that returns output of the network            
            out = obj.X_out;
            if isempty(out)
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
        
        function obj = eval(obj)
            obj.eval_mode = true;
        end
        
        function obj = train(obj)
            obj.eval_mode = false;
        end
        
        % BNLayer helpers
        function [gamma, beta] = getBNParam(obj)
            n = numel(obj.bn_layers);
            gamma = cell(n, 1);
            beta = cell(n, 1);
            for i = 1: n
                gamma{i} = obj.bn_layers{i}.gamma;
                beta{i} = obj.bn_layers{i}.beta;
            end
        end
        
        function obj = setBNParam(obj, gamma, beta)
            n = numel(obj.bn_layers);
            for i = 1: n
                obj.bn_layers{i}.gamma = gamma{i};
                obj.bn_layers{i}.beta = beta{i};
            end
        end
        
        function [grad_g, grad_b] = getBNGrad(obj)
            n = numel(obj.bn_layers);
            grad_g = cell(n, 1);
            grad_b = cell(n, 1);
            for i = 1: n
                grad_g{i} = obj.bn_layers{i}.grad_gamma;
                grad_b{i} = obj.bn_layers{i}.grad_beta;
            end
        end
        
    end
end