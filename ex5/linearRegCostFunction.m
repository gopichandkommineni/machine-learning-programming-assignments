function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h=X*theta;
k=size(theta,1);
p=(lambda/(2*m))*sum((theta(2:k).^2));
J=sum((h-y).^2)/(2*m)+p;


%grad(1)=sum((h-y)*X(1))/m;
%s=(lambda/m)*theta(2:k);
%grad(2:k)=sum((h-y).*X(:,2:k))/m;
%grad(2:k)=grad(2:k)+s;
temp=theta;
temp(1)=0;
grad=(1/m).*(X'*(h-y)+lambda.*temp);








% =========================================================================

grad = grad(:);

end
