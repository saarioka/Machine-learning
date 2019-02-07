function [mu,Sigma,p]=cifar_10_bayes_learn_COV(f,labels)
    data_dim = size(f,2);
    mu = nan(10,data_dim);
    Sigma = nan(10*data_dim,data_dim);
    p = nan(10,1);
    j = 1;
    for i = 1:10
        row = (labels == (i-1));
        f_right = f(row,:);
        mu(i,:) = mean(f_right,1);
        sig = cov(f_right);
        Sigma(j:(j+data_dim-1),:) = sig;
        p(i) = size(f_right,1)/size(f,1);
        j = j+data_dim;
    end
end