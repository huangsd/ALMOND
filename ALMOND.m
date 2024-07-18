% Sample-level Multi-view Graph Clustering
function [res,Loss, Loss1, Loss2, Loss3, Loss1_p1, Loss1_p2, ZV] = SLMVGC(data,labels, alpha, gamma, beta, rho, normData, max_iter)
% data: cell array, view_num by 1, each array is num_samp by d_v
% num_clus: number of clusters
% num_view: number of views
% num_samp: number of instances
% labels: groundtruth of the data, num by 1
% alpha: trade-off parameter for ||Z^(v)||_F^2
% beta : trade-off parameter for \sum_{i=1}^n ||S_i^T - W_i*(Z_hat)_i||_2^2
% gamma: trade-off parameter for ||S||_F^2
% rho: Lagrangian Multiplier
if nargin < 3
    alpha = 1;
end
if nargin < 4
    gamma = 1;
end
if nargin < 5
    beta = 1;
end
if nargin < 6
    rho = 1;
end
if nargin < 7
    normData = 1;
end
if nargin < 8
    max_iter = 5;
end

num_view = size(data,1);
num_samp = size(labels,1);
num_clus = length(unique(labels)); 
% === Normalization1 ===
if normData == 1
    for i = 1:num_view
        dist = max(max(data{i})) - min(min(data{i}));
        m01 = (data{i} - min(min(data{i})))/dist;
        data{i} = 2*m01 - 1;
    end
end
% === Normalization2 ===
if normData == 2
    for iter = 1:num_view
        data{iter} = normalize(data{iter});
    end
end
%

%  ====== Initialization =======
sumS = zeros(num_samp);

XV = data;  % data matrix
ZV = cell(num_view, 1);  % subspace matrix
Z_bar = zeros(num_samp, num_samp, num_view); % subspace tensor (by concatenating all ZVs to a tensor)
JV = cell(num_view, 1);    % Lagrangian Multiplier
J_bar = zeros(num_samp, num_samp, num_view); % Lagrangian Multiplier tensor (by concatenating all JVs to a tensor)
YV = cell(num_view, 1);
Y_bar = zeros(num_samp, num_samp, num_view);
S = zeros(num_samp, num_samp);  % final similarity matrix
A = ones(num_view, num_samp) / num_view;  % sample-level weight matrix
    
% === initialize Zv===
for v = 1:num_view 
    X = XV{v};
    Zv = (X*X' + alpha*eye(num_samp))\(X*X'); % initial Zv
    Zv = max(Zv,0);
    Zv = (Zv + Zv')/2;
    Zv = Zv - diag(diag(Zv));
    ZV{v} = Zv;
    sumS = sumS + Zv;
end

% === initialize Z_bar === 
for v = 1:num_view
    Z_bar(:,:,v) = ZV{v};
end

% === initialize J_bar and Y_bar===

for v = 1:num_view
    JV{v} = unifrnd(-1, 1, num_samp, num_samp);
    % JV{v} = rand(num_samp, num_samp) ;
    YV{v} = zeros(num_samp);
    Y_bar(:,:,v) = YV{v};
    J_bar(:,:,v) = JV{v};
end

% === initialize S ===
S = sumS/num_view;

% === initialize I ===
I = eye(num_samp);

% ================== iteration ==================
% fprintf('begin updating ......\n')
iter = 0;
bstop = 0;
Loss = zeros(max_iter+1,1);
Loss1 = zeros(max_iter+1,1);
Loss2 = zeros(max_iter+1,1);
Loss3 = zeros(max_iter+1,1);
Loss1_p1 = zeros(max_iter+1,1);
Loss1_p2 = zeros(max_iter+1,1);
% for iter = 1: Iter
while ~bstop
    iter = iter + 1;
%     fprintf('the %d -th iteration ......\n', iter);

    % === update Zv ===
    for v = 1:num_view
        Xv = XV{v};
        Jv = JV{v};
        Yv = YV{v};
        Zv = (2*(Xv*Xv') + (alpha + rho)*I) \ (rho*Jv + 2*(Xv*Xv') - Yv);
%         Zv = (2*(Xv*Xv') + (alpha + 1)*I) \ (1*Jv + 2*(Xv*Xv') );
%         A = (lambda + phi)*I + L;
%         Q = QV{v};
%         for i = 1:num_samp
%             index = find(GV{v}(i,:)>0);
%             Ii = I(i,index);
%             qi = Q(i,index);
%             b = 2*lambda*Ii + 2*phi*qi;
%             % solve z^T*A*z-z^T*b
%             [zi, ~] = fun_alm(A(index,index),b);
%             ZV{v}(i,index) = zi';
%         end
        Zv = max(Zv,0);
        Zv = Zv - diag(diag(Zv));
        ZV{v} = (Zv + Zv')/2;
    end
    %

    % === update Z_bar
    for v = 1:num_view
        Z_bar(:,:,v) = ZV{v};
    end
    %

    % === update J ===
    for i = 1:num_samp
        Yi = squeeze(Y_bar(i,:,:))';
%         Yi = zeros(num_view, num_samp);
%         for v = 1:num_view
%             Yi(v,:) = Y_bar(i,:,v);
%         end
        Ai = A(:,i);
        Zi = squeeze(Z_bar(i,:,:));
        Si = S(:,i);
        Ji = (rho*eye(num_view) + beta*(Ai*Ai'))\(rho*Zi' + beta*Ai*Si' + Yi);
        J_bar(i,:,:) = Ji';
    end
    for v = 1:num_view
        tmp = J_bar(:,:,v);
        tmp = (tmp + tmp')/2;
        tmp = max(0, tmp);
        tmp = tmp - diag(diag(tmp));
        J_bar(:,:,v) = tmp;
        JV{v} = J_bar(:,:,v);
    end
    
    %
    % === update S ===
    for i = 1:num_samp
        Ji = zeros(num_view, num_samp);
        for v = 1:num_view
            Ji(v, :) = J_bar(:, i, v)';
        end
        h = -(-beta*A(:,i)'*Ji)/(gamma + beta);
        S(i,:) = EProjSimplex_new(h);
    end
    S = (S + S')/2;
    %

    % === update rho ===
%     epslion = 0;
%     for v = 1:num_view
%         epslion = epslion + ZV{v} - JV{v};
%     end
%     if epslion > 10e-6
        rho = rho*1.21;
%     end
    %

    % === update A ===
        for i = 1:num_samp
            Ji = zeros(num_view, num_samp);
            for v = 1:num_view
                Ji(v, :) = J_bar(:, i, v)';
            end
            T = ones(num_view,1)*S(:,i)' - Ji;
            B = (T*T')\ones(num_view, 1);
            C = ones(1, num_view)/(T*T')*ones(num_view, 1);
            D = B / C;
            A(:, i) = D;
        end
    %
    
    % === Update Yv ===
    for m = 1:num_view
        YV{v} = YV{v} + rho*(ZV{v} - JV{v});
        Y_bar(:,:,v) = YV{v};
    end
    %

    L1_loss = 0; L2_loss = 0; L3_loss = 0; 
    for v=1:num_view
        Loss1_p1(iter) = Loss1_p1(iter) + norm(XV{v}' - XV{v}'*ZV{v},'fro')^2;
        Loss1_p1(isnan(Loss1_p1)) = 0;
        Loss1_p2(iter) = Loss1_p2(iter) + alpha/2*norm(ZV{v},'fro')^2;
        Loss1_p2(isnan(Loss1_p2)) = 0;
        L1_loss = L1_loss + Loss1_p1(iter) + Loss1_p2(iter);
    end
    
    for i = 1:num_samp
        Ji = zeros(num_view, num_samp);
        Zi = zeros(num_view, num_samp);
        for v = 1:num_view
            Ji(v, :) = J_bar(:, i, v)';
            Zi(v, :) = Z_bar(:, i, v)';
        end
        L2_loss = L2_loss + beta/2*norm(S(:,i)' - A(:,i)'*Ji,'fro')^2;
    end

    L3_loss = L3_loss + gamma/2*norm(S,"fro")^2;
    Loss(iter) = L1_loss + L2_loss + L3_loss; 
    Loss1(iter) = L1_loss;
    Loss2(iter) = L2_loss;
    Loss3(iter) = L3_loss;
%     if (iter > 1) && ((iter > max_iter) || (abs(Loss(iter-1)-Loss(iter))/Loss(iter-1) <= 1e-6))
%         bstop = 1;
%     end
    if (iter > 1) && ((iter > max_iter))
        bstop = 1;
    end
end
% x=1:1:iter;%x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
%  a=Loss; %a数据y值
%  plot(x,a,'r','LineWidth',2); %线性，颜色，标记
% axis([0,iter,-inf,inf])  %确定x轴框图大小
% ax = gca;
% ax.YAxis.Exponent = 4;
% legend('100Leaves');   %右上角标注
% xlabel('Iteration')  %x轴坐标描述
% ylabel('Loss') %y轴坐标描述


S(isnan(S)==1) = 0;
y = SpectralClustering(S, num_clus);
res = EvaluationMetrics(labels, y);
end



