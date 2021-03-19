function [diff]=MyGradientCheck(lambda)

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

%Setting up a mini neural network example
input_no=3;
hidden_no=5;
output_no=3;
m=5;
e=0.5;

%Initializing random theta values
theta1=(rand(hidden_no,input_no+1)*2*e)-e;
theta2=(rand(output_no,hidden_no+1)*2*e)-e;
theta_nn=[theta1(:);theta2(:)];

%Initializing random X and y
X=rand(m,input_no);
y=randi(output_no,m,1);

%Finding gradients using back propagation
[J_nn,grad_nn]=nnCostFunction(theta_nn,input_no,hidden_no,output_no,X,y,lambda);

%Finding gradients using central difference (CD)

%Constant for central difference
epsilon=10^-4;

%Initializing gradient vector and a reusable unrolled theta vector
grad_cd=zeros(size(grad_nn));
theta_cd=theta_nn;

%Looping over parameters to find gradient using CD
for i=1:numel(grad_nn)
    
    %J(theta+epsilon)
    theta_cd(i)=theta_cd(i)+epsilon;
    [J_plus,not_used]=nnCostFunction(theta_cd,input_no,hidden_no,output_no,X,y,lambda);
    
    %J(theta-epsilon)
    theta_cd(i)=theta_cd(i)-(2*epsilon);
    [J_minus,not_used]=nnCostFunction(theta_cd,input_no,hidden_no,output_no,X,y,lambda);
    
    grad_cd(i)=(J_plus-J_minus)/(2*epsilon);
    disp(J_plus-J_minus)
    theta_cd=theta_nn;
    
end

disp([grad_nn grad_cd]);
diff=(sum(grad_nn-grad_cd))/numel(grad_nn);

end