
% evaluate classifier with respect to uncertainlty estimation

function [correct, wrong] = evaluate(data,correct_labels)

correctly_classified = [];
wrongly_classified = [];

[m, n] = size(correct_labels);
for i=1:n
    index = correct_labels(:,i:i);
    correctly_classified = [correctly_classified; data(index:index, :)];
end

% find rows in data that are also in correctly_classified
commonRows = ismember(data,correctly_classified,'rows');

% remove those rows
data(commonRows,:) = [];
wrongly_classified = data;

% histogram for correctly classified
hist(correctly_classified,10);
%dp = (x(2)-x(1));
%area = sum(p)*dp;
%p = p/area;
%H = -sum( (p*dp) * log2(p) );

% histogram for wrongly classified
%[p1,x1] = hist(wrongly_classified,10);
hist(wrongly_classified,10);
%dp1 = (x1(2)-x1(1));
%area1 = sum(p1)*dp1;
%p1 = p1/area1;
%H1 = -sum( (p1*dp1) * log2(p1) );

correct = correctly_classified;
wrong = wrongly_classified;