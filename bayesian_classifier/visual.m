cifar_10_read_data;

X = double(tr_data);
Y = tr_labels;
tabulate(Y)

% Train a naive Bayes classifier. It is good practice to specify the class order.
Mdl = fitcnb(X,Y,...
    'ClassNames',{'0','1','2','3','4','5','6','7','8','9'});

% Plot the Gaussian contours.
figure;
gscatter(X(:,1),X(:,2),Y);
h = gca;
cxlim = h.XLim;
cylim = h.YLim;