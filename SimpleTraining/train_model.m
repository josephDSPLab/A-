test_data = 'energy_efficiency_cooling_load_testing.csv';
train_data = 'energy_efficiency_cooling_load_training.csv';
train_h_data = 'energy_efficiency_cooling_load_training_header.csv';
train = load(train_data);
test = load(test_data);
X = train(:,2:end)';
Y = train(:,1)';
net= mymlp([size(X,1),2,1],[],1);
alpha = 0.001;
for j = 1:1000
    err = 0;
    order = randperm(length(Y));
    for i = order
        out = net.forward(X(:,i));
        delt = out - Y(i);
        err = err + abs(delt);
        grad = net.backward(delt');
        net.optimize(alpha,0);
    end
    err
end