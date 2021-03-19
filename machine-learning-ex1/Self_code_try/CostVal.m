function [J]=CostVal(theta,X,y)
%To calculate the cost function!
m=length(y);
J=(1/(2*m))*sum(((X*theta)-y).^2);

end