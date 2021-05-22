function [net, smooth_loss] = adaGrad(X_vec, Y_vec, net)
% A function that performs AdaGrad for RNN.
% ----------
% Arguments:
%   X_vec: [bool] k (onehot) X n_data
%   Y_vec: [bool] k (onehot) X n_data
%   net: network to be trained
% Return:
%   net: trained network
%   smooth_loss: loss curve, n_step X 1

    % debug settings: print info every *** step
    PRINT_LOSS = 100;
    PRINT_TEXT = 500;
    
    % parameters for optimization algorithm
    global ADA RNN
    lr = 0.1;                           % learning rate
    n_epoch = ADA.n_epoch;              % #epoch
    len = net.seq_len;                  % sequence length
    n_seq = floor(size(X_vec, 2)/len);  % #seq each epoch
    n_step = n_seq * n_epoch;           % #step 
    smooth_loss = zeros(n_step, 1);     % smooth loss
    
    % init cumulative squared gradient
    for k = RNN.field_names
        par = k{1};
        G.(par) = zeros(size(net.(par)));
    end

    % --- epoch loop ---
    for i = 1: n_epoch
        % reset hidden state in each epoch
        net = net.clearHidden();
        
        % --- sequence loop ---
        for j = 1: n_seq
            
            iter = (i-1) * n_seq + j;  % index: iteration (step)
            
            % get sequence data
            X = X_vec(:, 1 + (j-1) * len : j * len);
            Y = Y_vec(:, 1 + (j-1) * len : j * len);
            
            % forward and backward propagation
            net = net.computeGradients(X, Y);
            P = net.output();
            
            % AdaGrad optimization
            for k = RNN.field_names
                par = k{1};  % string
                g = net.grad.(par);
                G.(par) = G.(par) + g .* g;
                net.(par) = net.(par) - lr ./sqrt(G.(par) + eps) .* g;
            end
            
            % compute loss and smooth_loss     
            loss = crossEntropy(P, Y);
            
            if iter == 1
                smooth_loss = loss;
            else          
                smooth_loss(iter) = 0.999 * smooth_loss(iter-1) + 0.001 * loss;
            end
            
            % print smooth_loss every 100 steps
            if ~mod(iter, PRINT_LOSS)
                fprintf("Step %d, smooth loss: %f\n", iter, smooth_loss(iter));
            end
            
            % synthesize text (200 chars) every 500 steps
            if ~mod(iter, PRINT_TEXT)
                fprintf("Step %d\n", iter);
                % TODO: 
            end
     
        end
        % --- END sequence loop ---
     
    end
    % --- END epoch loop ---
    
end