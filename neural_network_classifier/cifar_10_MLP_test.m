function estlabel = cifar_10_MLP_test(x,net)
    x = double(x');
    y = net(x);
    estlabel = vec2ind(y) - 1;
end