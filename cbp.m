function [xhat, Lxhat] = cbp(H, L)
% Classical belief propagation for decoding based on LLRs

    adj_vns = cell(size(H,1),1);
    adj_cns = cell(1,size(H,2));

    for i = 1:size(H,1)
        adj_vns{i} = find(H(i,:)==1);
    end
    for j = 1:size(H,2)
        adj_cns{j} = find(H(:,j)==1)';
    end

    deg1_vns = find(sum(H,1)==1);

    % BP decoding for all bits
    VN = H .* repmat(L,size(H,1),1);
    CN = H;

    rounds = 20;
    for r = 1:rounds
        % CN update
        VN(H==0) = Inf;
        T = zeros(size(CN));
        col_inds = 1:size(H,2);
        for i = 1:size(H,2)
            cols = setdiff(col_inds,i);
            T(:,i) = prod(tanh(VN(:,cols)/2),2) .* H(:,i);
        end
        CN = 2*atanh(T);
        
        % VN update
        s = sum(CN,1);
        S = repmat(s,size(H,1),1) .* H;
        VN = S - CN;
        for j = deg1_vns
            VN(adj_cns{j},j) = L(1,j);
        end
    end
    
    % Final VN update
    Lxhat = L + sum(CN,1);
    
    xhat = (Lxhat==0).*(rand(1,size(H,2))<0.5) + (Lxhat~=0).*double(Lxhat<0); 

end