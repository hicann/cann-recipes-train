import torch
import torch.nn as nn

# 输入: (B, T, D)，单头版本
def chunk_batched_delta_rule_forward(Q,K,V,beta,C,
                                    initial_state=None):
    """
        Q,K and V are of shape (B,L,d) where B is the batch, L the sequence length and d the embeding's dimension
        beta is of shape (B,L,1)
        C is the size of the chunk, it should divide L
    """
    B,L,d = Q.shape
    Q, K, V= map(lambda x: x.reshape(B,-1,C,d), [Q,K,V])
    beta = beta.reshape(B,-1,C)
    K_beta = K*beta.unsqueeze(-1)
    V_beta = V*beta.unsqueeze(-1)

    mask = torch.triu(torch.ones(C,C), diagonal=0).bool()

    K_t = torch.transpose(K,2,3)
    T = -(K_beta[:] @ K_t[:]).masked_fill(mask,0)
   
   #forward substitution 
    for k in range(L//C):
        for i in range(1,C):
            T_new = T.clone()
            T_new[:,k,i,:i] = T[:,k,i,:i] + (T[:,k,i,:,None]*T[:,k,:,:i]).sum(-2)
            T = T_new
        T[:,k] = T[:,k] + torch.eye(C)

    W = T @ K_beta
    U = T @ V_beta

    # 初始化 S
    if initial_state is not None:
        # 你的 h0 是 (B, H, D, D)，这里 H=1
        S = initial_state.squeeze(1)  # (B, d, d)
    else:
        S = torch.zeros((B, d, d), device=Q.device, dtype=Q.dtype)

    O = torch.empty_like(V)
    mask = torch.triu(torch.ones(C,C), diagonal=1).bool()

    for i in range(L//C):
        q_i, k_i, w_i = Q[:,i], K[:,i], W[:,i]
        u_i = U[:,i]-w_i@S
        o_inter = q_i @ S
        A_i = (q_i @ k_i.transpose(1,2)).masked_fill(mask,0)
        o_intra = A_i @ u_i
        S = S + k_i.transpose(1,2)@u_i
        O[:,i] = o_intra + o_inter

    final_state = S 
        
    return O.reshape(B,L,d), final_state



# 输入: (B, T, H, D)，多头版本
def chunk_batched_delta_rule_forward_multi(Q,K,V,beta,C,
                                    initial_state=None):
    """
        Q,K and V are of shape (B,L,d) where B is the batch, L the sequence length and d the embeding's dimension
        beta is of shape (B,L,1)
        C is the size of the chunk, it should divide L
    """
    device = Q.device
    B, L, H, D = Q.shape
    # 分块: (B, H, N, C, D)
    Q = Q.permute(0, 2, 1, 3).reshape(B, H, -1, C, D)  # (B, H, N, C, D)
    K = K.permute(0, 2, 1, 3).reshape(B, H, -1, C, D)
    V = V.permute(0, 2, 1, 3).reshape(B, H, -1, C, D)
    beta = beta.permute(0, 2, 1).reshape(B, H, -1, C)   # (B, H, N, C)
    
    # 扩展 beta 到 D 维（如果需要 per-dim beta，但通常 per-head）
    K_beta = K * beta.unsqueeze(-1)   # (B, H, N, C, D)
    V_beta = V * beta.unsqueeze(-1)

    mask = torch.triu(torch.ones(C,C), diagonal=0).bool().to(device)

    # 计算 A = -β K K^T （下三角）
    K_t = K.transpose(-1, -2)  # (B, H, N, D, C)
    A = -(K_beta @ K_t).masked_fill(mask, 0)  # (B, H, N, C, C)
   
   #forward substitution 
    T_mat = A.clone()
    for i in range(1, C):
        # Extract X[i, :i]  → shape (..., i)
        X_ik = T_mat[..., i, :i]  # (B, H, N, i)
        
        # Extract X[:i, :i] → shape (..., i, i)
        X_kj = T_mat[..., :i, :i]  # (B, H, N, i, i)
        
        # Compute sum_{k=0}^{i-1} X[i,k] * X[k,j] for j in [0, i)
        # (..., i, 1) * (..., i, i) → (..., i, i) → sum over k (dim=-2) → (..., i)
        correction = (X_ik.unsqueeze(-1) * X_kj).sum(dim=-2)  # (B, H, N, i)
        
        # 必须 out-of-place 更新！
        T_mat_updated = T_mat.clone()
        T_mat_updated[..., i, :i] = T_mat[..., i, :i] + correction
        T_mat = T_mat_updated  # 重新赋值，不修改原张量
    
    I = torch.eye(C, device=A.device).view(1, 1, 1, C, C).to(device)
    T_mat = T_mat + I

    # W = T @ K_beta, U = T @ V_beta
    W = T_mat @ K_beta  # (B, H, N, C, D)
    U = T_mat @ V_beta

    # === 初始化状态 S ===
    if initial_state is not None:
        S = initial_state  # (B, H, D, D)
    else:
        S = torch.zeros((B, H, D, D), device=Q.device, dtype=Q.dtype)
    
    
    O = torch.empty_like(V).to(device)  # (B, H, N, C, D)
    mask_intra = torch.triu(torch.ones(C, C, device=Q.device), diagonal=1).bool().to(device)

    N = Q.shape[2]

    for i in range(N):
        q_i = Q[:, :, i]      # (B, H, C, D)
        k_i = K[:, :, i]      # (B, H, C, D)
        w_i = W[:, :, i]      # (B, H, C, D)
        u_i = U[:, :, i]      # (B, H, C, D)


        u_prime = u_i - w_i @ S  # (B, H, C, D)

        o_inter = q_i @ S       # (B, H, C, D)

        A_i = (q_i @ k_i.transpose(-2,-1)).masked_fill(mask_intra, 0)  # (B, H, C, C)

        o_intra = A_i @ u_prime

        O[:, :, i] = o_intra + o_inter

        # 更新状态: S = S + k_i^T @ u_prime
        S = S + torch.einsum('bhci,bhcj->bhij', k_i, u_prime)     # (B, H, D, D)

    final_state = S
    # 恢复 shape: (B, H, N, C, D) -> (B, T, H, D)
    O = O.permute(0, 2, 1, 3, 4).reshape(B, L, H, D)
        
    return O, final_state
    


def delta_rule_recurrent_step(q_t, k_t, v_t, beta_t, S_prev):
    """
    Perform a single step of the recurrent Delta Rule.
    
    Args:
        q_t: Query vector at time step t, shape (d,).
        k_t: Key vector at time step t, shape (d,).
        v_t: Value vector at time step t, shape (d,).
        beta_t: Writing strength scalar at time step t, shape ().
        S_prev: Previous hidden state (memory matrix), shape (d, d).
        
    Returns:
        o_t: Output vector at time step t, shape (d,).
        S_new: Updated hidden state (memory matrix), shape (d, d).
    """
    # Compute old value
    v_old_t = S_prev @ k_t  # Shape (d,)
    
    # Compute new value
    v_new_t = beta_t * v_t + (1 - beta_t) * v_old_t  # Shape (d,)
    
    # Update hidden state (memory)
    S_new = S_prev - torch.outer(v_old_t, k_t) + torch.outer(v_new_t, k_t)  # Shape (d, d)
    
    # Compute output
    o_t = S_new @ q_t  # Shape (d,)
    
    return o_t, S_new

class DeltaBlock(nn.Module):
    def __init__(self,d,expand=1, neg_eigen=False):
        """
            d is the dimension of the input
            d*expand is the size of the hidden state
            neg_eigen if true allow the model to have negative eigen value. It was not on the original paper but on another: https://arxiv.org/abs/2411.12537.
        """
        super(DeltaBlock,self).__init__()
        self.d = d
        self.expand = expand
        self.Wq = nn.Linear(d,d*expand)
        self.Wk = nn.Linear(d,d*expand)
        self.Wv = nn.Linear(d,d*expand)

        self.proj_out = nn.Linear(d*expand,d)

        self.beta = nn.Linear(d,1)
        self.sigma = nn.Sigmoid()
        self.alpha = 2 if neg_eigen else 1

    def forward(self,X,chunk=1):
        """
            this is the chunkwise form of deltanet
            input: 
                X of shape B,L,d
                chunk size
            output: Y of shape B,L,d
        """
        if chunk ==1:
            _,chunk,_ = X.shape
        o, _ = chunk_batched_delta_rule_forward(
            self.Wq(X), self.Wk(X), self.Wv(X) / self.alpha,
            self.alpha * self.sigma(self.beta(X)), chunk,
        )
        return self.proj_out(o)

    def step(self,X,S=None):
        """
            this is the parallel form of deltanet
            input:
                X vector of shape d
                S state of shape (d,d), if not provided, the model will initialize it with zeros(0,0)
            output:
                Y vector of shape d
                S new state of shape (d,d)
        """
        if S==None:
            S = torch.zeros(self.d*self.expand,self.d*self.expand)
        y,S = delta_rule_recurrent_step(self.Wq(X),self.Wk(X),self.Wv(X)/self.alpha,self.alpha*self.sigma(self.beta(X)),S)
        return self.proj_out(y), S