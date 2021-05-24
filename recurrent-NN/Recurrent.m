classdef Recurrent
% Recurrent neural network.

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
        syn_len
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
            obj.syn_len = param.syn_len;
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
        
        function text = synthesize(obj, x0, h0, n_syn)
            % NOTE: Recurrent is a value class so this function
            %       does not effect the network when training.
            % --------------
            % x0: k X 1, first input char
            % h0: k X 1, first hidden state
            % n_syn: synthesis length
            
            if nargin < 4 || isempty(n_syn)
                n_syn = obj.syn_len;
            end
            
            if nargin < 3 || isempty(h0)
                h0 = zeros(obj.m, 1);
            end
            
            global DICT
            y_char = cell(1, n_syn);
            obj.h0 = h0;
            
            obj.H = zeros(obj.m, n_syn);
            obj.A = zeros(obj.m, n_syn);
            obj.O = zeros(obj.k, n_syn);
            obj.P = zeros(obj.k, n_syn);
            obj.X = zeros(obj.k, n_syn + 1);
            obj.X(:, 1) = x0;
            
            obj.A(:, 1) = obj.W * obj.h0 + obj.U * obj.X(:, 1) + obj.b;
            obj.H(:, 1) = tanh(obj.A(:, 1));
            obj.O(:, 1) = obj.V * obj.H(:, 1) + obj.c;
            obj.P(:, 1) = softmax(obj.O(:, 1));
            % get char
            [~, y_int] = max(obj.P(:, 1));
            y_char{1} = DICT.int2char(y_int);
                
            for t = 2: n_syn
                obj.A(:, t) = obj.W * obj.H(:, t-1) + obj.U * obj.X(:, t) + obj.b;
                obj.H(:, t) = tanh(obj.A(:, t));
                obj.O(:, t) = obj.V * obj.H(:, t) + obj.c;
                obj.P(:, t) = softmax(obj.O(:, t));
                
                % deterministic
                % [~, y_int] = max(obj.P(:, t));
                % y_char{t} = DICT.int2char(y_int);
                % obj.X(:, t+1) = DICT.coder.encode(y_int);
                
                % random
                cp = cumsum(obj.P(:, t));
                a = rand;
                ixs = find(cp-a>0);
                ii = ixs(1);
                y_char{t} = DICT.int2char(ii);
                obj.X(:, t+1) = ind2vec(ii, obj.k);   
            end
            
            text = strjoin(y_char, '');
        end
        
    end
end