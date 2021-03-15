clear
clc

addpath(genpath('utils'));

load('data4classification/indian_pines_randomSampling_0.1_run_1.mat');
y_tra = double(y_tra);
y_test = double(y_test);
[~, y_tra] = max(y_tra, [], 2);
[~, y_test] = max(y_test, [], 2);

load('results/drl_30_bands_indian_pines.mat');
selected_bands = selected_bands+1; % python -> matlab

x_tra = x_tra(:, selected_bands);
x_test = x_test(:, selected_bands);

Factor = ClassificationKNN.fit(x_tra, y_tra, 'NumNeighbors', 3);
y_pred = predict(Factor, x_test);

oa = length(find((y_pred-y_test) == 0))/length(y_test);

[kappa, ~, accu_class] = evaluatePerf(y_pred, y_test);
aa = mean(accu_class);