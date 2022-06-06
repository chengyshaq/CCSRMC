function [model_LSRLSF] = LSRLSF( X, Y, optmParameter)
   %% optimization parameters
    alpha            = optmParameter.alpha;
    gamma            = optmParameter.gamma;
    beta             = optmParameter.beta;
    theta            = optmParameter.theta;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;
   %% initializtion
    num_dim= size(X,2);
    num_labels=size(Y,2);
    num_N = size(X,1);
    XTX = X'*X;
    V_1 = causalInference(Y);
    V_2 = V_1';
    
    P_s   =  eye(num_labels);
    P_s_1 =P_s;
    S_s = eye(num_N,num_labels);
    S_s_1 = S_s;
    P = 1 - pdist2(Y'+eps,Y'+eps,'cosine');
%     L = diag(sum(P,2)) - P;
    [row, col] = find(V_2 == 1);
    loopSize = size(row);
    for i = 1 : loopSize
        P(row(i), col(i)) = 0;
    end
    % L = diag(sum(P, 2)) - P;
    L = P;
    iter    = 1;
    bk = 1;
    bk_1 = 1; 
    oldloss = 0;
   %% proximal gradient
    while iter <= maxIter

        norm_1=norm(XTX)^2; 
        W_s   = (XTX + theta*eye(num_dim)) \ (X'*S_s_1);
        W_s_1 = W_s;
        Lip = sqrt(2*(norm_1) + norm(alpha*L)^2);


       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
       
       Gw_s_k = W_s_k - 1/Lip * ((XTX*W_s_k - X'*S_s_1) + alpha*W_s_k*L);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       W_s    = softthres(Gw_s_k,beta/Lip);
       
       P_s_k = pinv(Y'*Y + 2*gamma*diag(sqrt(sum(P_s_1.^2,2))))*(Y'*S_s_1);
       P_s_1 = P_s_k;
       S_s_k = 1/(alpha + 3)*(Y + Y*P_s_1 + X*W_s);
       S_s_1 = S_s_k;
       predictionLoss = 0.5*trace((X*W_s - S_s_1)'*(X*W_s_1 - S_s_1))+0.5*alpha*trace(W_s*L*W_s');%
       subspaceLoss = 0.5*trace((Y*P_s_1 - S_s_1)'*(Y*P_s_1 - S_s_1)) + 0.5*trace((S_s_1 - Y)'*(S_s_1 - Y)) + 0.5*alpha*norm(S_s_1,'fro');%
       spares_Ps_Loss= sum(sqrt(sum(P_s_1.*P_s_1,2)),1);
       spares_Ws_Loss = sum(sum(W_s~=0));
       totalloss = predictionLoss + subspaceLoss + gamma*spares_Ps_Loss + beta*spares_Ws_Loss;

       if abs(oldloss - totalloss) <= miniLossMargin

           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       if iter>maxIter

       end
       iter=iter+1; 
    end

      model_LSRLSF.W=W_s;
      model_LSRLSF.S=S_s_1;
      model_LSRLSF.P=P_s_1;
end

%% soft thresholding operator 
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end
