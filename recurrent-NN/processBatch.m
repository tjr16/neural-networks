function processBatch
    global INFO RNN
    
    int_arr = arrayfun(@(x) INFO.char2idx(x), INFO.book_data);
    
    abandon =  mod(length(int_arr) ,RNN.seq_len);
    if abandon == 0
        abandon = abandon + RNN.seq_len;
    end
    
    int_X = int_arr(1: end-abandon);
    int_Y = int_arr(2: end-abandon+1);
    
    vec_X = full(ind2vec(double(int_X), RNN.k));  % k X n
    vec_Y = full(ind2vec(double(int_Y), RNN.k));  % k X n
    
%     % cell 1 X n, each cell: k X 1
%     cell_X = mat2cell(vec_X, [RNN.k], ones(1, size(vec_X, 2)));
%     cell_Y = mat2cell(vec_Y, [RNN.k], ones(1, size(vec_Y, 2)));
    
    save('info/vec_data.mat', 'vec_X', 'vec_Y');

end