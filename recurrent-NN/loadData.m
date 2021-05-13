function info = loadData(restart)
% A function that loads data from book txt file.
% ----------
% Argument:
%   restart: re-read all data and save info again
% Return:
%   info: [struct]

    if nargin < 1
        restart = false;
    end
    
    if restart
        book_fname = 'goblet_book.txt';
        fid = fopen(book_fname, 'r');
        book_info.book_data = fscanf(fid, '%c');
        fclose(fid);

        book_chars = unique(book_info.book_data);
        book_info.K = numel(book_chars);
        book_info.char = num2cell(book_chars);
        book_info.idx = int32(1:80);
        book_info.char2idx = containers.Map(book_info.char, book_info.idx);
        book_info.idx2char = containers.Map(book_info.idx, book_info.char);
        
        save('info/book_info.mat', 'book_info');
        info = book_info;   
    else    
        load('info/book_info.mat', 'book_info');
        info = book_info;
    end
 
end
