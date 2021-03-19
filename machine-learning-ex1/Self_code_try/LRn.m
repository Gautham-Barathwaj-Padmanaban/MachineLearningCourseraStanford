%%Basic multivariate linear regression using four different methods%%
clear all;clc; close all;
data=load('ex1data1.txt');
[m,n]=size(data);
x=data(:,(1:n-1));
X=[ones(m,1) data(:,(1:n-1))];
y=data(:,n);
%--------------------------GRADIENT DESCENT------------------------------%
%You can vary parameters for gradient descent here
iters=500;
alpha=0.01;

theta=zeros(size(X,2),1);
J=zeros(iters+1,1);
J(1)=CostVal(theta,X,y);

for i=2:iters+1
    J(i)=CostVal(theta,X,y);
    theta=GradDescent(theta,X,y,alpha);
end

iterp=0:1:iters;
figure;
plot(iterp,J);
xlabel('Number of iterations')
ylabel('J_\theta')
title('Cost Function for Gradient Descent')
disp(theta)

a=[1 min(x)-2;1 max(x)+2];
b=a*theta;
figure;
scatter(x,y);
hold on;
plot(a(:,2),b);
title('Gradient Descent')
xlabel('Population of city in 10000s')
ylabel('Profits of city in 10000s')
hold off;

%-------------------------NORMAL EQUATION--------------------------------%
theta=zeros(size(X,2),1);
theta=(pinv(X'*X))*X'*y;
a=[1 min(x)-2;1 max(x)+2];
b=a*theta;
figure;
scatter(x,y);
hold on;
plot(a(:,2),b);
title('Normal Equation')
xlabel('Population of city in 10000s')
ylabel('Profits of city in 10000s')
hold off;

%---------------------USING FMINUC quasi newton-----------------------------------%
options=optimset('GradObj','on','MaxIter',1000,'FunValCheck','on');
initialTheta=zeros(size(X,2),1);
[theta,J,exitflag,O]=fminunc(@(t)(costandgrad(t,X,y)),initialTheta,options);
a=[1 min(x)-2;1 max(x)+2];
b=a*theta;
figure;
scatter(x,y);
hold on;
plot(a(:,2),b);
title('Fminuc-quasi newton (default)')
xlabel('Population of city in 10000s')
ylabel('Profits of city in 10000s')
hold off;

%---------------------USING FMINUC trust region-----------------------------------%
options=optimset('GradObj','on','MaxIter',1000,'FunValCheck','on','Algorithm','trust-region');
initialTheta=zeros(size(X,2),1);
[theta,J,exitflag,O]=fminunc(@(t)(costandgrad(t,X,y)),initialTheta,options);
a=[1 min(x)-2;1 max(x)+2];
b=a*theta;
figure;
scatter(x,y);
hold on;
plot(a(:,2),b);
title('Fminuc trust region')
xlabel('Population of city in 10000s')
ylabel('Profits of city in 10000s')
hold off;









