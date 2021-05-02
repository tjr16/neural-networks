classdef BatchNormLayer
% A value class as a batch normalization layer.

    %% === PROPERTIES ===
    properties
        l_size    % layer size
        % parameter  
        gamma
        beta
        % distribution
        mean_val  % from forward propagation
        var_val
        mean_av   % average value from exponential moving average
        var_av
        av_init   % [bool] flag: mean_av and var_av initialized
        % forward: intermediary values
        s      % input bn layer
        s_hat  % after normalization
        s_out  % after scale & shift
        % backward: gradients
        grad_gamma
        grad_beta
        grad_s   % s = Wx+b
    end
   
    %% === METHODS ===
    methods
        function obj = BatchNormLayer(ls)
            % Constructor of class DoubleLayer
            % ls: layer size
            % ----------
            
            obj.l_size = ls;
            obj.gamma = ones(ls, 1);
            obj.beta = zeros(ls, 1);
            obj.av_init = false;
        end
        
        function obj = forward(obj, s, eval)
            global BN
            if eval  % evaluate: use empirical value
                assert(obj.av_init);
                mu = obj.mean_av;
                sigma2 = obj.var_av;
            else  % train: get value from data
                [obj.mean_val, obj.var_val] = BatchNormLayer.getNormal(s);
                mu = obj.mean_val;
                sigma2 = obj.var_val;
                if obj.av_init
                    obj.mean_av = BN.alpha * obj.mean_av + (1 - BN.alpha) * mu;
                    obj.var_av = BN.alpha * obj.var_av + (1 - BN.alpha) * sigma2;
                else
                    obj.mean_av = mu;
                    obj.var_av = sigma2;
                    obj.av_init = true;
                end
            end
            obj.s = s;
            obj.s_hat = BatchNormLayer.batchNormalize(s, mu, sigma2);
            obj.s_out = obj.gamma .* obj.s_hat + obj.beta;
        end
        
        function obj = computeGradient(obj, grad_in)
            % In training mode ...
            % grad_in: gradient back from dropout and activation
            nb = size(grad_in, 2);
            % Compute gradient for the scale and offset parameters
            obj.grad_gamma = (grad_in .* obj.s_hat) * ones(nb, 1)/ nb;
            obj.grad_beta = grad_in * ones(nb, 1)/nb;
            % Propagate gradient through the scale and shift
            G = grad_in .* (obj.gamma * ones(1, nb));  % n_layer X nb
            % Propagate G through batch normalization
            sig1 = 1./sqrt((obj.var_val + eps));
            sig2 = (obj.var_val + eps).^(-1.5);
            G1 = G .* (sig1 * ones(1, nb));
            G2 = G .* (sig2 * ones(1, nb));
            D = obj.s - obj.mean_val * ones(1, nb);
            c = (G2 .* D) * ones(nb, 1);
            obj.grad_s = G1 - (G1 * ones(nb, 1)) * ones(1, nb)/nb - ...
                D .* (c * ones(1, nb))/nb;
        end

    end
    
    %% === STATIC METHODS ===
    methods(Static)

        function [mu, sigma2] = getNormal(X)
        % This function computes arguments of a normal distribution.
        % ----------
        % Arguments:
        %   X: input data, each data as a column.
        %      feature_dim X data_amount
        % Return:
        %   mu: mean value of each dimension
        %   sigma2: variance of each dimension
        %       (biased estimation, 1/n)

            mu = mean(X, 2);
            sigma2 = var(X, 1, 2);  
        end
        
        function s_out = batchNormalize(s, mu, sigma2)
        % This function does batch normalization.
        % ----------
        % Arguments:
        %   s: input data
        %   mu: mean value of input data
        %   sigma2: variance of input data
        % Return:
        %   s_out: normalized

            s = s - mu;
            s_out = s ./ sqrt(sigma2 + eps);
        end
        
    end
end