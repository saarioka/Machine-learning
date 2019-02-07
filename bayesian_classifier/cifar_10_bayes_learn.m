function [mu,sigma,p]=cifar_10_bayes_learn(f,labels)
    mu = nan(10,size(f,2));
    sigma = mu;
    p = nan(10,1);
    for i = 1:10
        row = (labels == (i-1));
        f_right = f(row,:);
        mu(i,:) = mean(f_right,1);
        sigma(i,:) = sqrt(var(f_right));
        p(i) = size(f_right,1)/size(f,1);
    end
end