% accuracy using logistic regression
function [acc,thres] = lrAcc(x,y)
    n = size(x,1);
    b = mnrfit(x,y+1);
    % in the sigmoid function, we have sig(0)=0.5. 
    % because we want a threshold that corresponds to 0.5 probability, 
    % we have 1*b1+thres*b2=0,
    % thus, thres = -b1/b2.
    thres = -(b(1)/b(2));
    acc = sum(y==(x>=thres))/n;
%     % code below does the same thing
%     pred = 1-round(logsig([ones(n,1) x]*b));
%     acc = sum(pred==y)/n;
end

