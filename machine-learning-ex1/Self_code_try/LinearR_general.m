%%Basic multivariate linear regression using four different methods%%

clear all;clc; close all;
data=load('ex1data1.txt');
[m,n]=size(data);
x=data(:,(1:n-1));
X=[ones(m,1) data(:,(1:n-1))];
y=data(:,n);

options=optimset('GradObj','on','MaxIter',1000,'FunValCheck','on','Algorithm','trust-region');
initialTheta=zeros(size(X,2),1);
[theta,J,exitflag,O]=fminunc(@(t)(Costfn(t,X,y)),initialTheta,options);










