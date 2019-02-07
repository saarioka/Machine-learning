function c = cifar_10_bayes_classify_COV(f,mu,Sigma,p)
    data_dim = size(f,2);
    c = nan(length(f),1);
    posteriori = nan(1,10);
    for i = 1:size(f,1)
        j = 1;
        for cl = 1:10
            sig = Sigma(j:(j+data_dim-1),:);
            % Make sure Sigma is a valid covariance matrix
            [~,err] = cholcov(sig,0);
            if err ~= 0
                disp('bad covariance')
%                 % Try a Moore-Penrose pseudoinverse
%                 sig = pinv(sig);
                  [V,D] = eig(sig);  
                sig = V*max(D,0)/V;
                posteriori(cl) = mvnpdf(f(i,:),mu(cl,:),sig)*p(cl);
            else
                posteriori(cl) = mvnpdf(f(i,:),mu(cl,:),sig)*p(cl);
            end
            j = j+data_dim;
        end
        [~,ind] = max(posteriori);
        c(i) = ind - 1;
    end
end