# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import torch.optim as optim

# class BERTConfig:
#     """ BERT configuration """
#     n_layer = 12
#     n_head = 12
#     embed_dim = 768
#     hidden_dim = 3072
#     max_position_embeddings = 512
#     p = 0.2
#     vocab_size = 30522
#     pad_token_id = 0

# # Planning the flow of the code  
# '''
# NEEDED TO DO :
# 1. Self Attention Block
# 2. Bert Block (this can be repeated again and again)
# '''

# class SelfAttention(nn.Module):
#     """ Self-attention block """
#     def __init__(self, embed_dim, n_head):
#         super(SelfAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.n_head = n_head
#         self.head_dim = embed_dim // n_head

#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
    
#     def transform(self, x, layer):
#         """ Transform input x using the specified layer (query, key, or value) """
#         B, T, C = x.size()
#         projection = layer(x)
#         projection = projection.view(B, T, self.n_head, self.head_dim)
#         projection = projection.transpose(1, 2) # B, n_head, T, head_dim
#         return projection
    
#     def attention(self, key, query, value, attention_mask):
#         """ Compute attention scores """
#         B, n_head, T, head_dim = key.size()
#         K_T = key.transpose(-2,-1) # B, n_head, head_dim, T
#         scores = query @ K_T / (head_dim ** 0.5) # B, n_head, T, T
#         scores = scores + attention_mask
#         attn_wts = F.softmax(scores, dim=-1)
#         output = attn_wts @ value # B, n_head, T, head_dim
#         output = output.transpose(1,2).contiguous().view(B,T,-1) # B, T, embed_dim
#         return output

#     def forward(self, x, attention_mask):
#         """ Forward pass for self-attention block """
#         Q = self.transform(x, self.query)
#         K = self.transform(x, self.key)
#         V = self.transform(x, self.value)
#         attention_output = self.attention(K, Q, V, attention_mask)
#         return attention_output
    
# class BERT_BLOCK(nn.Module):
#     """ BERT block with self-attention and feed-forward network """
#     def __init__(self, config):
#         super(BERT_BLOCK, self).__init__()
#         self.attention_block = SelfAttention(config.embed_dim, config.n_head)
#         self.layer_norm = nn.LayerNorm(config.embed_dim)
#         self.interm_linear_layer = nn.Linear(config.embed_dim, config.hidden_dim)
#         self.out_linear_layer = nn.Linear(config.hidden_dim, config.embed_dim)
#         self.dropout = nn.Dropout(config.p)
#         self.linear_layer = nn.Linear(config.embed_dim, config.embed_dim)
#         self.activation = nn.GELU()
    
#     def add_norm_block(self, input_prev, output_prev, layer):
#         """ Add layer norm and resudual connection """
#         output = input_prev + self.dropout(layer(output_prev))
#         output = self.layer_norm(output)
#         return output
    
#     def forward(self, x, attention_mask):
#         attention_output = self.attention_block(x, attention_mask)
#         out = self.add_norm_block(x, attention_output, self.linear_layer)
#         ffn = self.activation(self.interm_linear_layer(out))
#         output = self.add_norm_block(out, ffn, self.out_linear_layer)
#         return output
    

# class BERT(nn.Module):
#     """ BERT model with multiple BERT blocks """
#     def __init__(self, config):
#         super(BERT, self).__init__()
#         self.config = config
#         self.bert_blocks = nn.ModuleList([BERT_BLOCK(config) for _ in range(config.n_layer)])

#         self.word_embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
#         self.position_embed = nn.Embedding(config.max_position_embeddings, config.embed_dim)
#         self.dropout = nn.Dropout(config.p)
#         self.layer_norm = nn.LayerNorm(config.embed_dim)

#         # for cls token
#         self.linear_cls = nn.Linear(config.embed_dim, config.embed_dim)
#         self.activation_cls = nn.GELU()
    
#     def forward(self, input_ids, attention_mask):
#         """ Forward pass for BERT model """
#         B, T = input_ids.size()
#         word_embeddings = self.word_embed(input_ids)
#         position_ids = torch.arange(self.config.max_position_embeddings).unsqueeze(0)
#         pos_ids = position_ids[:,:T]
#         position_embeddings = self.position_embed(pos_ids)
#         embedding = word_embeddings + position_embeddings
#         embedding = self.dropout(embedding)
#         embedding = self.layer_norm(embedding)

#         extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#         extended_mask = (1.0 - extended_mask) * -10000.0

#         for block in self.bert_blocks:
#             embedding = block(embedding, extended_mask)

#         first_token = embedding[:,0]
#         first_token = self.linear_cls(first_token)
#         first_token = self.activation_cls(first_token)

#         return {'last_hidden_state': embedding, 'pooler_output': first_token}



# if __name__ == "__main__":
#     # Initialize BERT model with configuration
#     config = BERTConfig()
#     model = BERT(config)
    
#     # Set random seed for reproducibility
#     torch.manual_seed(42)
    
#     # Define random input shapes
#     batch_size = 2
#     seq_len = 16
    
#     # Generate random input_ids and attention_mask
#     input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len))
#     attention_mask = torch.ones(batch_size, seq_len)
    
#     # Forward pass
#     try:
#         outputs = model(input_ids, attention_mask)
#         print("No errors during forward pass!")
#         print("last_hidden_state shape:", outputs['last_hidden_state'].shape)
#         print("pooler_output shape:", outputs['pooler_output'].shape)
#     except Exception as e:
#         print("Error during forward pass:", e)


import numpy as np

# Define 5PL parameters (example values)
A = 1.5
D = 0.2
C = 2.0
B = 1.3
G = 0.9

def five_pl(x, A, D, C, B, G):
    """5-parameter logistic function y(x)"""
    return D + (A - D) / ( (1 + (x/C)**B )**G )

def five_pl_inverse(y, A, D, C, B, G):
    """Inverse 5PL: x(y)"""
    # Avoid division by zero or invalid roots
    if y == D:
        raise ValueError("y cannot be equal to D")
    base = (A - D) / (y - D)
    if base <= 0:
        raise ValueError("Invalid y for real solution")
    inner = base ** (1/G) - 1
    if inner < 0 and B % 2 == 0:
        raise ValueError("No real solution for even B and negative inner value")
    return C * (inner) ** (1/B)

# Test for a range of x values
x_values = np.linspace(0.1, 5, 10)  # Avoid x=0 to prevent division by zero
tolerance = 1e-6
success = True



for x in x_values:
    y = five_pl(x, A, D, C, B, G)
    try:
        x_recovered = five_pl_invesrse(y, A, D, C, B, G)
        if not np.isclose(x, x_recovered, atol=tolerance):
            print(f"Sanity check failed: x={x}, x_recovered={x_recovered}")
            success = False
    except Exception as e:
        print(f"Exception for x={x}, y={y}: {e}")
        success = False

if success:
    print("Sanity check passed: The conversion is correct!")
else:
    print("Sanity check failed for some values.")

