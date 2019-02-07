%% Ex3
cifar_10_read_data;

%% Task 1
f_tr = cifar_10_features(tr_data); % opetus
f_te = cifar_10_features(te_data); % testi
[mu,sigma,p] = cifar_10_bayes_learn(f_tr,tr_labels); % opeta
c = cifar_10_bayes_classify(f_te,mu,sigma,p); % testaa
result = cifar_10_evaluate(c,te_labels); % tulokset
disp(['Accuracy for 1-dimensional npdf' num2str(result)])

%% Task 2 & 3
sizes = [32 16 8 4 2 1];
accuracy = [];
for part_size = sizes
%     tic
    f_tr = cifar_10_features(tr_data, part_size);
    f_te = cifar_10_features(te_data, part_size);
    [mu,sigma,p] = cifar_10_bayes_learn_COV(f_tr,tr_labels);
    c = cifar_10_bayes_classify_COV(f_te,mu,sigma,p);
    result = cifar_10_evaluate(c,te_labels);
%     toc
    disp(['Accuracy for size ' num2str(part_size)  ': ' num2str(result)]);
    accuracy = [accuracy result];
end

figure;
    scatter(sizes,accuracy,'filled');
    title('Accuracy vs sub-window size\newline(multivariate normal distribution)');
    xlabel('Partition size (pixels)');
    ylabel('Accuracy (%)');
    axis([0 35 0 100]); grid on;
    text(sizes-1,accuracy-4,string(accuracy));
  