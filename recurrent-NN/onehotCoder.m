classdef onehotCoder
% Onehot encoding and decoding.

    properties
        n_cls
    end

    methods
      
        function obj = onehotCoder(k)
            obj.n_cls = k;
        end
        
        function vec = encode(obj, ind)
            % input (ind) must be a row vector or cell[row vector]
            
            vec = full(ind2vec(ind, obj.n_cls));  % n_cls X length(ind)
        end
        
        function ind = decode(obj, vec)
            ind = vec2ind(vec);
        end
    
    end
end