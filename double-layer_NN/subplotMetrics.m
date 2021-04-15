function subplotMetrics(metrics, show_step, simple)
% A function that plots loss, cost and accuracy curves in one figure.
% ----------
% Arguments:
% 	metrics: [loss; loss_val; cost; cost_val; acc; acc_val];
%   show_step: int, how many steps in one epoch
%       if using #epoch as x-axis, set this argument to 0 or neglect it.
%   simple: bool, using simple curve or not
    
    if nargin < 3
        simple = true;
    end
    
    if nargin < 2
        show_step = 0;  % do not change the scale of x-axis
    end
    
    if simple
        blue = 'b';
        red = 'r';
    else
        blue = '-bo';
        red = '-rx';
    end
    
    loss_train = metrics(1, :); loss_valid = metrics(2, :);
    cost_train = metrics(3, :); cost_valid = metrics(4, :);
    acc_train = metrics(5, :); acc_valid = metrics(6, :);

    subplot(2, 2, 1);
    if show_step
        plot(1: show_step: show_step * numel(loss_train), loss_train, blue);
        hold on;
        plot(1: show_step: show_step * numel(loss_valid), loss_valid, red);
        xlbl = 'update step';
    else
        plot(loss_train, blue);
        hold on;  
        plot(loss_valid, red);
        xlbl = 'epoch';
    end 
    legend('training', 'validation');
    xlabel(xlbl); ylabel('loss'); title('Mean Loss');

    subplot(2, 2, 2);
    if show_step
        plot(1: show_step: show_step * numel(cost_train), cost_train, blue);
        hold on;
        plot(1: show_step: show_step * numel(cost_valid), cost_valid, red);
    else
        plot(cost_train, blue);
        hold on;  
        plot(cost_valid, red);
    end 
    legend('training', 'validation');
    xlabel(xlbl); ylabel('cost'); title('Cost');
    
    subplot(2, 2, 3);
    if show_step
        plot(1: show_step: show_step * numel(acc_train), acc_train, blue);
        hold on;
        plot(1: show_step: show_step * numel(acc_valid), acc_valid, red);
    else
        plot(acc_train, blue);
        hold on;  
        plot(acc_valid, red);
    end
    legend('training', 'validation');
    xlabel(xlbl); ylabel('acc'); title('Accuracy');

end