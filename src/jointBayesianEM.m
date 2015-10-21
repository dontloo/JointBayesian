% estimate within and between covariances using EM
% 1. init with estimated within and between (theta
% 2. MLE p(x|theta) using EM till converge
% if epoch is set to 0, will return then estimated between and within covs
% directly. if thre is set to 0, EM will terminate only when enough number
% of epoches have been excuted.
% initializations of between and within covs have to be positive semidefinite,
% in order to satisfy the properties of a cov matrix.
% in order to randomly initialize a positve semidefinite matrix,
% we can first randomly initialize a matrix, then compute its convariance,
% because covariance matrices are always positive semi-definite.
function [A,G,S_mu,S_eps] = jointBayesianEM(train_x, train_lbl, epoch, thres, feature_dim, dat_num, sub_num)
    x_cell = cell(sub_num,1);
    buff_size = 1;% maximum number of samples (pictures) per person
    for i=1:sub_num
        x_cell{i} = train_x(:,train_lbl==i);
        buff_size = max([buff_size size(x_cell{i},2)]);
    end
    num_buff = zeros(buff_size,1); % num_buff(m)==1 means there's at least one subject with m samples
    for i=1:sub_num
        num_buff(size(x_cell{i},2))=1;
    end
    
    % init with matrices used in LDA
    data_mean = mean(train_x,2); % should be zero if already subtracted
    S_mu = zeros(feature_dim); % identity matrix are positive definite
    S_eps = zeros(feature_dim);
    single_num = 0; % number of subjects with only one sample
    for i=1:sub_num
        n_k = size(x_cell{i},2);
        m_k = mean(x_cell{i},2);        
        
        % within class covariance matrix
        if n_k>1
            tmp = bsxfun(@minus,x_cell{i},m_k); % tmp = x-m (prml eq 4.43
            S_eps = S_eps + tmp*tmp'; % prml eq 4.43
%             % result is identical with the matrix ops above
%             S_k = zeros(feature_dim);
%             for k=1:n_k
%                 S_k = S_k + (x_cell{i}(:,k)-m_k)*(x_cell{i}(:,k)-m_k)';
%             end
        else
            single_num = single_num + 1;
        end
        
        % between class covariance matrix
        S_mu = S_mu + n_k*(m_k-data_mean)*(m_k-data_mean)'; % prml eq 4.46
    end
    S_mu = S_mu/dat_num; % have to devide dat_num to stay in the same magnitude as the EM approach
    S_eps = S_eps/(dat_num-single_num);
    
%     % init to be positive definite
%     S_mu = eye(feature_dim); % identity matrix are positive definite
%     S_eps = eye(feature_dim);
    
    mu = zeros(feature_dim,sub_num);
    eps = zeros(feature_dim,dat_num);
    S_mu_FmG = cell(buff_size,1);
    S_eps_G = cell(buff_size,1);
    for z=1:epoch        
        fprintf('\nepoch: %i\n',z);
        % E step (independent of data actually
        % pre calculate terms to be used the M step
        F = inv(S_eps); % eq 5
        for m = 1:buff_size
            if(num_buff(m)==1)
                fprintf('%i',m);            
                G = -((m*S_mu + S_eps)\S_mu)/S_eps; % eq 6
                S_mu_FmG{m} = S_mu*(F+m*G); % term of eq 7
                S_eps_G{m} = S_eps*G; % term of eq 8
            end
        end

        % M step 
        eps_ctr = 0;
        for i = 1:sub_num
            m = size(x_cell{i},2);
            mu(:,i) = S_mu_FmG{m}*sum(x_cell{i},2); % eq 7
            eps(:,eps_ctr+1:eps_ctr+m) = bsxfun(@plus,x_cell{i},S_eps_G{m}*sum(x_cell{i},2)); % eq 8
            eps_ctr = eps_ctr + m;
        end
        old_S_mu = S_mu;
        old_S_eps = S_eps;
        S_mu = cov(mu');
        S_eps = cov(eps');
        % check for convergence
        % todo: check likelihood
        fprintf('\n%f\t%f\n',norm(S_eps - old_S_eps)/norm(S_eps),norm(S_mu - old_S_mu)/norm(S_mu));
        if norm(S_eps - old_S_eps)/norm(S_eps)<thres && norm(S_mu - old_S_mu)/norm(S_mu)<thres
            break;
        end
    end
    F = inv(S_eps); % eq 5
    G = -((2*S_mu + S_eps)\S_mu)/S_eps; % eq 6 (note that m==2 for comparing two imgs
    A = inv(S_mu + S_eps) - (F + G); % eq 5
end