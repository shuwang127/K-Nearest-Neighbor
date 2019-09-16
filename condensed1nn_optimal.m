%% Implemented condensed 1-NN classification for letter recognition data set.
%  Shu Wang
%

clear;
clc;

%% Set the parameters
filename = 'letter-recognition.data';
num = 20000;
dim = 16;
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
if ~exist('condense1.mat')
    cond_list = zeros(num_train, 1);
	dist = inf * ones(num_train, 1);
    est_train = zeros(num_train, 1);
    acc = 0;
    cnt = 0;
    disp('==========Condensed 1-NN training==========');
    while(true)
        % randomly select a sample which is incorrectly classified.
        ind_diff = find( (label_train - est_train) ~= 0 );
        ind_diff_rand = ind_diff( randperm( length(ind_diff) ) );
        ind_diff_slct = ind_diff_rand(1);
        % classify the training samples.
		data_new = data_train(ind_diff_slct, :);
		label_new = label_train(ind_diff_slct, :);
		dist_new = sum(((data_train - repmat(data_new, num_train, 1)) .^ 2), 2);
        dist_tmp = dist;
        est_train_tmp = est_train;
		for i = 1 : num_train
			if dist_new(i) < dist_tmp(i)
				dist_tmp(i) = dist_new(i);
				est_train_tmp(i) = label_new;
			end
		end
        % determine if the condense is completed.
        loss = sum(abs(label_train - est_train_tmp)); %
        acc_tmp = sum( (label_train - est_train_tmp) == 0 ) / num_train;
        num_err = num_train - sum( (label_train - est_train_tmp) == 0 );
        cnt = cnt + 1;
        if (acc_tmp > acc) || ((cnt > num_err) && (acc_tmp == acc))
            % generate condensed set.
            cond_list(ind_diff_slct) = 1;
            dist = dist_tmp;
            est_train = est_train_tmp;
            acc = acc_tmp;
            disp(['>>> Cond Num: ', num2str(sum(cond_list)), ', Loss: ', num2str(loss), ', Acc: ', num2str(100*acc), '% ',...
            '(', num2str(sum((label_train - est_train) == 0)), '/', num2str(num_train), ') '...
            'Loop Cnt: ', num2str(cnt)]);
            cnt = 0;
        end
        % Skip the loop
        if 1 == acc
            disp(['========Total ', num2str(sum(cond_list)), ' condensed samples========']);
            label_cond = label_train( find(cond_list == 1), : );
            data_cond = data_train( find(cond_list == 1), : );
            num_cond = length(label_cond);
			save condense1.mat data_cond label_cond num_cond cond_list;
            break;
        end
    end
else
    load condense1.mat;
end

%% Test processing
output_test = zeros(num_test, 1);
for i = 1 : num_test
    dist = sum(((data_cond - repmat(data_test(i, :), num_cond, 1)) .^ 2), 2);
    [~, ind] = sort(dist);
    ind_NN = ind(1);
    output_test(i) = label_cond(ind_NN);
end

%% Estimate
diff_test = output_test - label_test;
acc = sum(diff_test == 0) / num_test;
disp(['Condensed kNN algorithm with K = 1']);
disp(['The testing accuracy: ', num2str(acc*100), '%']);