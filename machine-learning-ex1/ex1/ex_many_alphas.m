clear ; close all; clc
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];
figure;
hold on
for i=0.01:0.1:1
    alpha = i;
    num_iters = 400;
    theta = zeros(3, 1);
    disp(theta)
    [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
    plot(1:numel(J_history),J_history,'-');
    xlabel('Number of iterations')
    ylabel('Cost function')
    legend()
end

