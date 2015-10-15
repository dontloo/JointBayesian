clear all;
% load training data
raw_data_dir = '../../../JointBayesianFeature/';
load([raw_data_dir 'lbp_WDRef.mat']);
load([raw_data_dir 'id_WDRef.mat']);
train_lbl = id_WDRef;
train_x = double(lbp_WDRef);

%train_x = sqrt(train_x);
train_mean = mean(train_x,1);
train_x = bsxfun(@minus,train_x,train_mean);%subtract the mean
[coeff,score,~] = pca(train_x);%PCA
dim_pca = 100;
train_x = score(:,1:dim_pca)';

clear id_WDRef;
clear lbp_WDRef;
clear score;

% load test data
load([raw_data_dir 'lbp_lfw.mat']);
load([raw_data_dir 'pairlist_lfw.mat']);
test_x = double(lbp_lfw);
test_intra = pairlist_lfw.IntraPersonPair;
test_extra = pairlist_lfw.ExtraPersonPair;

%test_x = sqrt(test_x);
test_mean = mean(test_x,1);
test_x = bsxfun(@minus,test_x,test_mean);
test_x = (test_x*coeff(:,1:dim_pca))';%PCA

clear lbp_lfw;
clear pairlist_lfw;
clear coeff;