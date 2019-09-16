%% Implemented condensed 1-NN classification for letter recognition data set.
%  Shu Wang
%

clear;
clc;

%% Set the parameters
filename = 'letter-recognition.data';
num = 20000;
dim = 16;
K = 1;
num_train = 15000;
num_test = num - num_train;
label = zeros(num, 1);
data = zeros(num, dim);

%% Read the dataset from file
if ~exist('letter-recognition.mat')
    file = textread(filename, '%s', 'delimiter', ',', 'whitespace', '');
    file = reshape(file, [dim+1, num])';
    for i = 1 : num
        label(i) = file{i, 1} - 'A' + 1;
        for j = 1 : dim
            data(i, j) = str2num( file{i, j+1} );
        end
    end
    clear file i j;
    save letter-recognition.mat data label;
else
    load letter-recognition.mat;
end

%% Divide the training set and testing set
data_train = data(1:num_train, :);
data_test = data((num_train+1):num, :);
label_train = label(1:num_train);
label_test = label((num_train+1):num);
clear data label;

%% Condensed processing
if ~exist('condense.mat')
    cond_list = zeros(num_train, 1);
    est_train = zeros(num_train, 1);
    disp(['==========Condensed k-NN training==========']);
    while(true)
        % randomly select a sample which is incorrectly classified.
        ind_diff = find( (label_train - est_train) ~= 0 );
        ind_diff_rand = ind_diff( randperm( length(ind_diff) ) );
        ind_diff_slct = ind_diff_rand(1);
        % generate condensed set.
        cond_list(ind_diff_slct) = 1;
        label_cond = label_train( find(cond_list == 1), : );
        data_cond = data_train( find(cond_list == 1), : );
        num_cond = length(label_cond);
        % classify the training samples.
        for i = 1 : num_train
            dist = sum(((data_cond - repmat(data_train(i, :), num_cond, 1)) .^ 2), 2);
            [~, ind] = sort(dist);
            ind_NN = ind(1:K);
            label_NN = label_cond(ind_NN);
            est_train(i) = mode(label_NN);
        end
        % determine if the condense is completed.
        loss = sum(abs(label_train - est_train)); %
        acc = sum( (label_train - est_train) == 0 ) / num_train;
        disp(['>>> Cond Num: ', num2str(sum(cond_list)), ', Loss: ', num2str(loss), ', Acc: ', num2str(100*acc), '% ',...
            '(', num2str(sum((label_train - est_train) == 0)), '/', num2str(num_train), ')']);
        if 1 == acc
            disp(['========Total ', num2str(sum(cond_list)), ' condensed samples========']);
            break;
        end
    end
    save condense.mat data_cond label_cond num_cond cond_list;
else
    load condense.mat;
end

%% Test processing
output_test = zeros(num_test, 1);
for i = 1 : num_test
    dist = sum(((data_cond - repmat(data_test(i, :), num_cond, 1)) .^ 2), 2);
    [~, ind] = sort(dist);
    ind_NN = ind(1:K);
    label_NN = label_cond(ind_NN);
    output_test(i) = mode(label_NN);
end

%% Estimate
diff_test = output_test - label_test;
acc = sum(diff_test == 0) / num_test;
disp(['Condensed kNN algorithm with K = ', num2str(K)]);
disp(['The testing accuracy: ', num2str(acc*100), '%']);
