function net = cifar_10_MLP_train(tr_data,tr_labels)
    labels = full(ind2vec(double(tr_labels)'+1));
    net = patternnet([50 50]);
    
    disp('Input weights before training:')
    inputw = net.IW;
    celldisp(inputw)
    
    disp('Layer weights before training:')
    layerw = net.LW;
    celldisp(layerw)
    
    disp('Biases before training:')
    biases = net.b;
    celldisp(biases)

    net = train(net,double(tr_data)',labels);
    
    disp('Input weights after training:')
    inputw = net.IW;
    celldisp(inputw)
    
    disp('Layer weights after training:')
    layerw = net.LW;
    celldisp(layerw)
    
    disp('Biases after training:')
    biases = net.b;
    celldisp(biases)
end