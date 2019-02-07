function f = cifar_10_features(x, n)
    if nargin == 1
        fr = mean(x(:,1:1024),2);
        fg = mean(x(:,1025:2048),2);
        fb = mean(x(:,2049:3072),2);
        f = [fr fg fb];
    else
        f = [];
        part = 32*n;
        for i = 1:part:(3072-part)
            f = [f mean(x(:,i:(i+part)),2)];
        end
        f = [f mean(x(:,(end-part):end),2)];


%         for i = 1:(32/n)
%             select_cols = [repmat(zeros(n,1),i-1,1); ones(n,1);
%                 repmat(zeros(n,1),32/n-i,1)];
%             select_cols = repmat(select_cols,32*3,1);
%             x_sub = mean(x*select_cols./size(select_cols,1));
%             f = [f x_sub];
            
%         for start = 1:n:32
%             x_sub = [];
%             for row = 1:n
%                 x_sub = x(:,start:(start+n));
%             end
%             f = [f x_sub];
%         end
    end
end