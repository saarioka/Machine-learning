% testing the evaluator

cifar_10_read_data;

acc_te = cifar_10_evaluate(te_labels,te_labels);
fprintf('Accuracy for training data %s%%\n', num2str(acc_te));

% random evaluator
rand_labels = [];
for data_ind = 1:size(tr_data,1)
    data_sample = tr_data(data_ind,:);
    result_label = cifar_10_rand(data_sample);
    rand_labels = [rand_labels; result_label];
end
acc_rand = cifar_10_evaluate(rand_labels,tr_labels);
fprintf('Accuracy for random classifier is %s%%\n', num2str(acc_rand));

% 1NN evaluator
NN_labels = [];
tr_data = int16(tr_data);
te_data = int16(te_data);
% for data_ind = 1:size(te_data,1)

% tic
for data_ind = 1:100
    data_sample = te_data(data_ind,:);
    result_label = cifar_10_1NN(data_sample);
%     result_label = cifar_10_1NN(data_sample,tr_data,tr_labels);
    NN_labels = [NN_labels; result_label];
end
% toc

acc_NN = cifar_10_evaluate(NN_labels,te_labels(1:100));
fprintf('Accuracy for nearest neighbour is %s%%\n', num2str(acc_NN));