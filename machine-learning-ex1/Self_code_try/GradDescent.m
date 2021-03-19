function [theta_new]=GradDescent(theta,X,y,alpha)
%Calculating the gradients using gradient descent

m=length(y);
theta_new=theta-((alpha/m)*(X'*((X*theta)-y)));

end
