function [net, smooth_loss, MIN] = adaGrad(X_vec, Y_vec, net)
% A function that performs AdaGrad for RNN.
% ----------
% Arguments:
%   X_vec: [bool] k (onehot) X n_data
%   Y_vec: [bool] k (onehot) X n_data
%   net: network to be trained
% Return:
%   net: trained network
%   smooth_loss: loss curve, n_step X 1
%   MIN: information of minimum smooth loss

    % debug settings: print info every *** step
    PRINT_LOSS = 500;
    PRINT_TEXT = 2000;
    
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
    
    % save min values
    MIN.loss = 1e8;
    MIN.iter = 1;
    % MIN.model

    % --- epoch loop ---
    for i = 1: n_epoch
        % reset hidden state in each epoch
        net = net.clearHidden();
        
        % --- sequence loop ---
        for j = 1: n_seq
            
            h0 = net.h0;  % save h0 for synthesis
            
            iter = (i-1) * n_seq + j;  % index: iteration (step)
            
            % get sequence data
            X = X_vec(:, 1 + (j-1) * len : j * len);
            Y = Y_vec(:, 1 + (j-1) * len : j * len);
            
            % forward and backward propagation
            net = net.computeGradients(X, Y);
            P = net.output();
            
            % AdaGrad optimization
            net = net.clipGradients();  % clip grad
            for k = RNN.field_names
                par = k{1};  % string
                g = net.grad.(par);
                G.(par) = G.(par) + g .^ 2;
                net.(par) = net.(par) - lr * g ./sqrt(G.(par) + eps);
            end
            
            % compute loss and smooth_loss     
            loss = crossEntropy(P, Y);
            
            if iter == 1
                smooth_loss(iter) = loss;
            else          
                smooth_loss(iter) = 0.999 * smooth_loss(iter-1) + 0.001 * loss;
            end
            
            if (smooth_loss(iter) < MIN.loss) && (smooth_loss(iter) < 55)
                MIN.loss = smooth_loss(iter);
                MIN.iter = iter;
                MIN.model = net;
            end
            
            % print smooth_loss every PRINT_LOSS steps
            if ~mod(iter, PRINT_LOSS)
                fprintf("Step %d, smooth loss: %f\n", iter, smooth_loss(iter));
            end
            
            % synthesize text (200 chars) every PRINT_TEXT steps
            if (iter == 1) || (~mod(iter, PRINT_TEXT))
                disp('--------');
                fprintf("Step %d\n", iter);
                text = net.synthesize(X(:, 1), h0);
                disp(text);
                disp('--------');
            end
     
        end
        % --- END sequence loop ---
     
    end
    % --- END epoch loop ---
    
end