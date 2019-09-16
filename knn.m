%% Implemented basic k-NN classification for letter recognition data set.
%  Shu Wang
%

clear;

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

%% k-NN
NN = zeros(num_test, num_train); %
output_test = zeros(num_test, 1);
for i = 1 : num_test
    dist = sum(((data_train - repmat(data_test(i, :), num_train, 1)) .^ 2), 2);
    [~, ind] = sort(dist);
    ind_NN = ind(1:K);
    label_NN = label_train(ind_NN);
    NN(i, :) = label_train(ind); %
    output_test(i) = mode(label_NN);
end

%% Estimate
diff_test = output_test - label_test;
acc = sum(diff_test == 0) / num_test;
disp(['KNN algorithm with K = ', num2str(K)]);
disp(['The testing accuracy: ', num2str(acc*100), '%']);

%% Determine the optimized K
disp('========Determine the optimal K========');
for kk = 1 : 10
    output_test = mode(NN(:, 1:kk), 2);
    diff_test = output_test - label_test;
    acc = sum(diff_test == 0) / num_test;
    disp(['KNN algorithm with K = ', num2str(kk)]);
    disp(['The testing accuracy: ', num2str(acc*100), '%']);
end
