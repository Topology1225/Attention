import math

import tqdm
import torch
import lightning
import torchsummary
from dataset import Dataset

class PositionalEncoding(torch.nn.Module):

  def __init__(self, dim, dropout = 0.1, max_len = 5000):
    super().__init__()
    self.dropout = torch.nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, 1, dim)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
    
  def forward(self, x):
    x = x + self.pe[:x.size(0)] # sin, cosを足す
    return self.dropout(x)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, head_num, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.linear_Q = torch.nn.Linear(dim, dim, bias=False)
        self.linear_V = torch.nn.Linear(dim, dim, bias=False)
        self.linear_K = torch.nn.Linear(dim, dim, bias=False)
        self.linear   = torch.nn.Linear(dim, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        
    def split_head(self, x):
        x = torch.tensor_split(x, self.head_num, dim=-1)
        x = torch.stack(x, dim=1) # bs, head_num, word_count/head_num, dim
        return x

    def concat_head(self, x):
        x = torch.tensor_split(x, x.size()[1], dim=1)
        x = torch.concat(x, dim=3).squeeze(dim=1)
        return x
        
    def forward(self, Q, K, V, mask=None):
        Q = self.linear_Q(Q)
        K = self.linear_K(K)
        V = self.linear_V(V)
        
        Q = self.split_head(Q) # bs, head_num, word_count/head_num, dim
        K = self.split_head(K) # bs, head_num, word_count/head_num, dim
        V = self.split_head(V) # bs, head_num, word_count/head_num, dim
        
        QK = torch.matmul(Q, torch.transpose(K, 3, 2)) # (bs, head_num, word_count/head_num, word_count/head_num)
        QK = QK/((self.dim // self.head_num) ** 0.5)
        
        if mask is not None:
            QK = QK + mask.unsqueeze(1) # ここ、maskする箇所だけinfにするとか必要じゃない？
            # nn.transfomer.generate_square_subsequent_mask()で-infの入ったmaskを作成してくれる！
            
        weight = torch.nn.functional.softmax(QK, dim=-1) # (bs, head_num, word_count/head_num)
        weight = self.dropout(weight)
        QKV = torch.matmul(weight, V) # (bs, head_num, word_count/head_num, dim)
        QKV = self.concat_head(QKV)
        QKV = self.linear(QKV)
        return QKV
        

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim=2048, dropout=0.1):
        super().__init__()
        
        self.dropout = torch.nn.Dropout(dropout)
        self.linear1 = torch.nn.Linear(dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderBlock(torch.nn.Module):
        def __init__(self, dim, head_num, dropout=0.1):
            super().__init__()
            self.multihead_attention = MultiHeadAttention(dim, head_num)
            self.layer_norm_1 = torch.nn.LayerNorm([dim])
            self.layer_norm_2 = torch.nn.LayerNorm([dim])
            self.feadforward  = FeedForward(dim)
            self.dropout_1    = torch.nn.Dropout(dropout)
            self.dropout_2    = torch.nn.Dropout(dropout)
            
        def forward(self, x):
            Q = K = V = x
            x = self.multihead_attention(Q, K, V)
            x = self.dropout_1(x)
            x = x + Q
            x = self.layer_norm_1(x)
            _x = x # residual 
            x = self.feadforward(x)
            x = self.dropout_2(x)
            x = x + _x # residual
            x = self.layer_norm_2(x)
            return x
            

class Encoder(torch.nn.Module):
    def __init__(self, encoder_vocabrary_size, dim, head_num, dropout=0.1):
        """
        Inputs:
            dim: int
                embeded dimention
            encoder_vocabrary_size: int
                for 
        """
        super().__init__()
        self.dim = dim 
        self.embedding = torch.nn.Embedding(encoder_vocabrary_size, dim)
        self.positional_ecoding = PositionalEncoding(dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.encoder_blocks = torch.nn.ModuleList(
            [
                EncoderBlock(dim, head_num) for _ in range(6)
            ]
        )
    
    def forward(self, japanese_token):
        x = self.embedding(japanese_token)
        x = x * (self.dim ** 0.5)
        x = self.positional_ecoding(x)
        x = self.dropout(x)
        for encoder in self.encoder_blocks:
            x = encoder(x)
        return x
    
class DecoderBlock(torch.nn.Module):
    def __init__(self, dim, head_num, dropout=0.1):
        super().__init__()
        self.masked_multihead_attention = MultiHeadAttention(dim, head_num)
        self.multihead_attention = MultiHeadAttention(dim, head_num)
        self.layer_norm_1 = torch.nn.LayerNorm([dim])
        self.layer_norm_2 = torch.nn.LayerNorm([dim])
        self.layer_norm_3 = torch.nn.LayerNorm([dim])
        self.feedforward = FeedForward(dim)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.dropout_3 = torch.nn.Dropout(dropout)
            
        
    def forward(self, x, encoder_output, mask):
        # masked encode for decoder input
        Q = K = V = x
        x = self.masked_multihead_attention(Q, K, V, mask)
        x = self.dropout_1(x)
        x = x + Q
        x = self.layer_norm_1(x)
        
        # multihead attention for xatten between encoder_output and decoder_input
        Q = x
        K = V = encoder_output
        x = self.multihead_attention(Q, K, V)
        x = self.dropout_2(x)
        x = x + Q # x + sublayer(x)
        x = self.layer_norm_2(x)
        
        # residual and feedforward
        _x = x # residual
        x = self.feedforward(x)
        x = self.dropout_3(x)
        x = x + _x # residual
        x = self.layer_norm_3(x)
        return x
        
        

class Decoder(torch.nn.Module):
    def __init__(self, decoder_vocabrary_size, dim, head_num, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.embedding = torch.nn.Embedding(decoder_vocabrary_size, dim)
        self.positional_embedding = PositionalEncoding(dim)
        self.decoder_blocks = torch.nn.ModuleList([
            DecoderBlock(dim, head_num) for _ in range(6)
        ])
        self.dropout = torch.nn.Dropout(dropout)
        self.linear  = torch.nn.Linear(dim, decoder_vocabrary_size)
    
    def forward(self, decoder_input, encoder_output, mask):
        x = self.embedding(decoder_input)
        x = x * (self.dim ** 0.5)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        for layer in self.decoder_blocks:
            x = layer(x, encoder_output, mask)
        x = self.linear(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, encoder_vocabrary_size, decoder_vocabrary_size, dim, head_num):
        super().__init__()
        self.encoder = Encoder(encoder_vocabrary_size, dim, head_num)
        self.decoder = Decoder(decoder_vocabrary_size, dim, head_num)
    
    def forward(self, batch):
        encoder_output = self.encoder(batch["japanese_token"])
        output = self.decoder(
            batch["decoder_input"],
            encoder_output,
            batch["mask"]
        )
        return output

        

def main():
    # define variables
    device = torch.device("mps")
    
    # define dataset
    dataset = Dataset(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0)
    
    # define criterion
    criterion = torch.nn.CrossEntropyLoss()

    # define model
    dict_vocab_size = dataset.get_vocab_size()
    model = Model(
        encoder_vocabrary_size=dict_vocab_size["japanese"],
        decoder_vocabrary_size=dict_vocab_size["english"],
        dim=512,
        head_num=8
    )
    model = model.to(device)
    
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_loss = 0
    for epoch in range(10):
        model.train()
        train_loss = 0
        for idx, batch in enumerate(dataloader):
            output = model(batch)
 
            target = torch.nn.functional.one_hot(batch["decoder_output"], dict_vocab_size["english"]).float().to(device)
            loss = criterion(output, target)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            print(f"epoch: {epoch} iteration: {idx} loss: {train_loss/(idx+1):.5f}")
            

if __name__=="__main__":
    main()
    
    