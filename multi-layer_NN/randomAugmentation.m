function X_new = randomAugmentation(X)
% A function that augments cifar-10 data (online).
     
    I = reshape(X, 32, 32, 3, []);
    size_I = size(I);
    num_I = size_I(end);

    for i = 1: num_I
        seed = rand;
        if seed > 2/3
            I(:, :, :, i) = fliplr(I(:, :, :, i));
        elseif seed > 1/3
            I(:, :, :, i) = imgaussfilt(I(:, :, :, i), 1);
        else
            I(:, :, :, i) = jitterColorHSV(...
            I(:, :, :, i), 'Contrast',0.05, 'Hue',0.015, ...
            'Saturation',0.025, 'Brightness',0.035);
        end
    end
    
    X_new = reshape(I, 3072, []);
   
end
