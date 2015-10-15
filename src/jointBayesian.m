% todo:
% 1. init with LDA
% 2. decompose G ( not really necessary
% 3. fit decision boundary
% 4. convergence check
% 5. other magics in the paper

clear all;
data_dir = '../../../data/JointBayesian/';
load([data_dir 'WDRef_pca_100.mat']);
feature_dim = size(train_x,1);
dat_num = size(train_x,2);
sub_num = max(train_lbl); % number of subjects (assume id number increases consectively

% EM
epoch = 100;
thres = 1e-6; % convergence threshold
[A,G,S_mu,S_eps] = jointBayesianEM(train_x,train_lbl,epoch,thres,feature_dim,dat_num,sub_num);
Sig_i = [S_mu+S_eps S_mu; S_mu S_mu+S_eps];
Sig_e = [S_mu+S_eps zeros(size(S_mu)); zeros(size(S_mu)) S_mu+S_eps];
% todo: fit a decision boundary
% test
% todo: decompose positive definite
test_pairs = [test_intra; test_extra];
test_lbl = [ones(size(test_intra,1),1);zeros(size(test_extra,1),1)];
test_r = zeros(size(test_lbl));
test_data_num = size(test_pairs,1);
for i=1:test_data_num
        test_r(i) = computeR(A,G,test_x(:,test_pairs(i,1)),test_x(:,test_pairs(i,2)));
end
% accuracy ( 100d: 0.8047 thres:-3.5
thres = -3.5;
acc = sum(test_lbl==(test_r>thres))/test_data_num;

% % logistic regression not so good
% b = mnrfit(test_r,test_lbl+1);
% pred = 1-round(logsig([test_r ones(test_data_num,1)]*b));
% acc = sum(pred==test_lbl)/test_data_num;

% probabilities too close to zero
% mvnpdf([test_x(:,test_pairs(3000,1));test_x(:,test_pairs(300,2))],zeros(2*dim_feature,1),Sig_i)