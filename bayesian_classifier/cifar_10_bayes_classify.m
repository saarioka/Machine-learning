function c = cifar_10_bayes_classify(f,mu,sigma,p)
    c = nan(length(f),1);
    posteriori = nan(1,10);
    for i = 1:size(f,1)
        for cl = 1:10
            posteriori(cl) = normpdf(f(i,1),mu(cl,1),sigma(cl,1))*...
                             normpdf(f(i,2),mu(cl,2),sigma(cl,2))*...
                             normpdf(f(i,3),mu(cl,3),sigma(cl,3))*...
                             p(cl);
        end
        [~,ind] = max(posteriori);
        c(i) = ind - 1;
    end
end