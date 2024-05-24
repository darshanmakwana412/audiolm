import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, cfg, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(cfg["emb_dim"]))
        self.shift = torch.nn.Parameter(torch.zeros(cfg["emb_dim"]))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["emb_dim"], cfg["mlp_hidd_dim"]),
            torch.nn.GELU(),
            torch.nn.Linear(cfg["mlp_hidd_dim"], cfg["emb_dim"])
        )
        
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Wq = torch.nn.Linear(cfg["emb_dim"], cfg["n_heads"]*cfg["head_dim"], bias=cfg["qkv_bias"])
        self.Wk = torch.nn.Linear(cfg["emb_dim"], cfg["n_heads"]*cfg["head_dim"], bias=cfg["qkv_bias"])
        self.Wv = torch.nn.Linear(cfg["emb_dim"], cfg["n_heads"]*cfg["head_dim"], bias=cfg["qkv_bias"])
        self.dropout = torch.nn.Dropout(cfg["drop_rate"])
        self.out_proj = torch.nn.Linear(cfg["n_heads"] * cfg["head_dim"], cfg["emb_dim"])
        self.register_buffer('mask', torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1))
    
    def forward(self, x):
        B, T, C = x.shape
        
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)
        
        queries = queries.view(B, T, self.cfg["n_heads"], self.cfg["head_dim"])
        keys = keys.view(B, T, self.cfg["n_heads"], self.cfg["head_dim"])
        values = values.view(B, T, self.cfg["n_heads"], self.cfg["head_dim"])
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        
        mask_bool = self.mask.bool()[:T, :T]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / self.cfg["head_dim"] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        context_vec = context_vec.contiguous().view(B, T, self.cfg["n_heads"] * self.cfg["head_dim"])
        context_vec = self.out_proj(context_vec)

        return context_vec

class Block(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.mlp = MLP(cfg)
        self.ln_1 = LayerNorm(cfg)
        self.ln_2 = LayerNorm(cfg)
        self.dropout = torch.nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x

class GPT(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])
        
        self.blocks = torch.nn.Sequential(*[Block(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg)
        # The unembedding matrix Wu
        self.out_head = torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)
        
        if cfg["verbose"]:
            self.print_info()
        
    def forward(self, x):
        B, T = x.shape
        
        tok_embs = self.tok_emb(x)
        pos_embs = self.pos_emb(torch.arange(T, device=x.device))
        x = tok_embs + pos_embs
        x = self.drop_emb(x)
        
        x = self.blocks(x)
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -self.cfg["context_length"]:]
            
            logits = self(idx_cond)
                
            logits = logits[:, -1, :] / temperature
            
            if top_k != None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probas = torch.softmax(logits, dim=-1)
            
            ## Greedy Sampling
#             idx_next = torch.argmax(probas, dim=-1, keepdim=True)

            idx_next = torch.multinomial(probas, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    
    def print_info(self):
        
        print(f"Initialzed a new model with config:")
        for conf, val in self.cfg.items():
            print(f"     {conf}: {val}")        

        # Calculate the total parameters in the model
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters in the model: {total_params/1e6:.2f} Million")
        
        # Assume float32 which takes 4 bytes a patameter
        total_model_size = total_params * 4 / (1024 * 1024)
        print(f"Total size of the model: {total_model_size:.2f} MB")
