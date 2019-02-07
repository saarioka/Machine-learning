%% Exercise 4: MLP

cifar_10_read_data;

%%
tr_data = cifar_10_features(tr_data,8);
neuralNet = cifar_10_MLP_train(tr_data,tr_labels);

%%
te_data = cifar_10_features(te_data,8);
estimated = cifar_10_MLP_test(te_data, neuralNet);

%%
acc = cifar_10_evaluate(te_labels', estimated);
% view(neuralNet)