classdef Recurrent
% A text version of recurrent neural network.

    properties
        % parameters
        W
        U
        V
        b
        c
        % intermediary values
        A
        H
        O
        P
        X
        h0
        % size
        m
        k
        seq_len
        % grad
        grad  % fieldnames: O, A, W, U, V, b, c
        % other
        field_names   % only names of parameters
    end

    methods
      
        function obj = Recurrent(param)           
            obj.U = param.U;
            obj.W = param.W;
            obj.V = param.V;
            obj.b = param.b;
            obj.c = param.c;
            
            obj.m = param.m;
            obj.k = param.k;
            obj.seq_len = param.seq_len;
            obj.field_names = param.field_names;
            
            obj.H = [];  % h0 set to 0 if H empty
        end

        function obj = initHidden(obj)            
            if isempty(obj.H)
                obj.h0 = zeros(obj.m, 1);
            else
                obj.h0 = obj.H(:, end);
            end
        end
        
        function obj = clearHidden(obj)
            % Clear hidden state for next epoch
            obj.H = [];
            obj.h0 = zeros(obj.m, 1);
        end
        
        function obj = forward(obj, X)
            % NOTE: input and output are onehot vectors
            
            obj = obj.initHidden();
            
            obj.H = zeros(obj.m, obj.seq_len);
            obj.A = zeros(obj.m, obj.seq_len);
            obj.O = zeros(obj.k, obj.seq_len);
            obj.P = zeros(obj.k, obj.seq_len);
            obj.X = X;  % k X seq_len
            
            obj.A(:, 1) = obj.W * obj.h0 + obj.U * obj.X(:, 1) + obj.b;
            obj.H(:, 1) = tanh(obj.A(:, 1));
            obj.O(:, 1) = obj.V * obj.H(:, 1) + obj.c;
            obj.P(:, 1) = softmax(obj.O(:, 1));
                
            for t = 2: obj.seq_len
                obj.A(:, t) = obj.W * obj.H(:, t-1) + obj.U * obj.X(:, t) + obj.b;
                obj.H(:, t) = tanh(obj.A(:, t));
                obj.O(:, t) = obj.V * obj.H(:, t) + obj.c;
                obj.P(:, t) = softmax(obj.O(:, t));
            end        
        end
        
        function obj = computeGradients(obj, X, Y)
            % Init hidden state to zero or last state;
            % Forward propagation;
            % Backward propagation
              
            obj = obj.forward(X);
            
            % grad output
            obj.grad.O = (obj.P - Y)';  % seq_len X k
            
            % grad linear A and hidden H
            obj.grad.H = zeros(obj.seq_len, obj.m);
            obj.grad.A = zeros(obj.seq_len, obj.m);
            obj.grad.H(end, :) = obj.grad.O(end, :) * obj.V;
            obj.grad.A(end, :) = obj.grad.H(end, :) * diag(1-tanh(obj.A(:, end)).^2);

            for t = obj.seq_len-1 : -1 : 1 
                obj.grad.H(t, :) = obj.grad.O(t, :) * obj.V + obj.grad.A(t+1, :) * obj.W; 
                obj.grad.A(t, :) = obj.grad.H(t, :) * diag(1-tanh(obj.A(:, t)).^2);
            end
            
            obj.grad.V = obj.grad.O' * obj.H';
            obj.grad.W = obj.grad.A' * [obj.h0 obj.H(:, 1:end-1)]';
            obj.grad.U = obj.grad.A' * X';
            obj.grad.b = obj.grad.A' * ones(obj.seq_len, 1);
            obj.grad.c = obj.grad.O' * ones(obj.seq_len, 1);   
        end
        
        function obj = clipGradients(obj)
            % Clip gradients
            
            for f = obj.field_names
                obj.grad.(f{1}) = max(min(obj.grad.(f{1}), 5), -5);
            end
        end
        
        function out = output(obj)
            % NOTE: input and output are onehot vectors
            out = obj.P;
        end
        
    end
end