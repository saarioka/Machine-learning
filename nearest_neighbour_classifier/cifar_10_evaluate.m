function acc = cifar_10_evaluate(pred,gt)
    pred = double(pred);
    gt = double(gt);
    same = pred(pred == gt);
    acc = length(same)/length(pred)*100;
    disp(['Accuracy is ' num2str(acc) '%']);
end