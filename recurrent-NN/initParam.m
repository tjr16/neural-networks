function initParam()
% A function that initialize RNN parameters.
% Some basic parameters should have been set in global struct RNN:
% -- requirements: RNN.m, RNN.k, RNN.sig
  
    global RNN
    RNN.b = zeros(RNN.m, 1);    % (a) bias vector
    RNN.c = zeros(RNN.k, 1);    % (b) bias vector
    RNN.U = randn(RNN.m, RNN.k) * RNN.sig ;  % (a) input weight
    RNN.W = randn(RNN.m, RNN.m) * RNN.sig ;  % (a) hidden weight
    RNN.V = randn(RNN.k, RNN.m) * RNN.sig ;  % (b) hidden weight
   
end