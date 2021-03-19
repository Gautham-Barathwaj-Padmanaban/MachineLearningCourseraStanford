function [J,gradient]=Costfn(theta,X,y)
%Cost function to be mimimized with gradients when using fminuc
m=length(y);
J=(1/(2*m))*sum(((X*theta)-y).^2);
gradient=(X'*((X*theta)-y));    

end
