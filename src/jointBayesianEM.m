function [A,G,S_mu,S_eps] = jointBayesianEM(train_x, train_lbl, epoch, thres, feature_dim, dat_num, sub_num)
    x_cell = cell(sub_num,1);
    buff_size = 1;% maximum number of pictures per person
    for i=1:sub_num
        x_cell{i} = train_x(:,train_lbl==i);
        buff_size = max([buff_size size(x_cell{i},2)]);
    end
    % todo: init with LDA
    % init to be positive definite
    S_mu = eye(feature_dim); % identity matrix are positive definite
    S_eps = eye(feature_dim);
    
    mu = zeros(feature_dim,sub_num);
    eps = zeros(feature_dim,dat_num);
    for z=1:epoch        
        fprintf('\nepoch: %i\n',z);
        % E step (independent of data actually
        % pre calculate terms to be used the M step
        S_mu_FmG = cell(buff_size,1);
        S_eps_G = cell(buff_size,1);
        for m = 1:buff_size
            fprintf('%i',m);
            F = inv(S_eps); % eq 5
            G = -((m*S_mu + S_eps)\S_mu)/S_eps; % eq 6
            S_mu_FmG{m} = S_mu*(F+m*G); % term of eq 7
            S_eps_G{m} = S_eps*G; % term of eq 8
        end

        % M step 
        eps_ctr = 1;
        for i = 1:sub_num
            m = size(x_cell{i},2);
            mu(:,i) = S_mu_FmG{m}*sum(x_cell{i},2); % eq 7
            eps(:,eps_ctr:eps_ctr+m-1) = bsxfun(@plus,x_cell{i},S_eps_G{m}*sum(x_cell{i},2)); % eq 8
            eps_ctr = eps_ctr + m;
        end
        old_S_mu = S_mu;
        old_S_eps = S_eps;
        S_mu = cov(mu');
        S_eps = cov(eps');
        % check for convergence
        % todo: check likelihood
        fprintf('\n%f\n',norm(S_eps - old_S_eps)/norm(S_eps));
        if norm(S_eps - old_S_eps)/norm(S_eps)<thres        
            break;
        end
    end
    F = inv(S_eps); % eq 5
    G = -((2*S_mu + S_eps)\S_mu)/S_eps; % eq 6 (note that m==2 for comparing two imgs
    A = inv(S_mu + S_eps) - (F + G);%formula£¨5£©
end