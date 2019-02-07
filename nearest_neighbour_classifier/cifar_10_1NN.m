function label = cifar_10_1NN(x)
    global tr_data tr_labels;
%     diffs = sum(sqrt((x - tr_data).^2),2);
    diffs = sum(abs(x - tr_data),2);
    [~, ind] = min(diffs);
    label = tr_labels(ind);
end