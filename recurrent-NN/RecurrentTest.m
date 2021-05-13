classdef RecurrentTest
% A text version of recurrent neural network.

    properties
        W
        U
        V
        b
        c
        % ---
        A
        H
        O
        P
        X
        % ---
        m
        k
        seq_len
    end

    methods
      
        function obj = RecurrentTest(param)           
            obj.U = param.U;
            obj.W = param.W;
            obj.V = param.V;
            obj.b = param.b;
            obj.c = param.c;
            
            obj.m = param.m;
            obj.k = param.k;
            obj.seq_len = param.seq_len;
            
        end

        function obj = randomGenerate(obj, X, n)
            % NOTE: input and output are onehot vectors
            
            obj.H = cell(n+2,1);
            obj.A = cell(n+2,1);
            obj.O = cell(n+2,1);
            obj.P = cell(n+2,1);
            obj.X = cell(n+2,1);
            
            obj.H{1} = zeros(obj.m, 1);  % h0
            obj.X{2} = X;   % input x1
            
            for t = 2: n + 1
                obj.A{t} = obj.W * obj.H{t-1} + obj.U * obj.X{t} + obj.b;
                obj.H{t} = tanh(obj.A{t});
                obj.O{t} = obj.V * obj.H{t} + obj.c;
                obj.P{t} = softmax(obj.O{t});               
                cp = cumsum(obj.P{t});
                a = rand;
                ixs = find(cp-a>0);
                ii = ixs(1);
                obj.X{t+1} = ind2vec(ii, obj.k);        
            end        
        end
        
        function out = output(obj)
            % NOTE: input and output are onehot vectors
            out = obj.X(2: end-1);
        end
        
    end
end