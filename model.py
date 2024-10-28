import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    The InputEmbeddings module converts input token indices into embeddings.

    Args:
        d_model (int): The dimension of the embeddings (embedding size).
        vocab_size (int): The number of distinct tokens (vocabulary size) that the model can process.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # nn.Embedding creates a lookup table that maps each token index to a vector of size d_model.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the input embeddings.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len), containing token indices.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model), containing scaled embeddings.
        """
        # Multiply by sqrt(d_model) to scale the embeddings.
        # This scaling helps to maintain appropriate magnitude of the embeddings relative to positional encodings.
        # Without scaling:
        # - Initial embedding values may be too small if initialized randomly.
        # - Positional encodings could dominate the embeddings when added together.
        # With scaling:
        # - Balances the magnitude of embeddings and positional encodings.
        # - Improves training stability and model performance.

        # The square root of d_model normalizes the variance of the embeddings.
        # It ensures that the embeddings have a reasonable range of values,
        # preventing them from becoming too small or too large as they propagate through the network.

        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
        Positional Encoding module injects information about the relative or absolute position
        of the tokens in the sequence. It is crucial for the Transformer model to capture the
        sequence structure since it processes tokens in parallel.

        Args:
            d_model (int): The dimension of the embeddings (embedding size).
            seq_len (int): The maximum length of the input sequences.
            dropout (float): Dropout rate to apply after adding positional encoding to embeddings.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch - explanation/implementation of positional encoding with pytorch
        # https://ai.stackexchange.com/questions/41670/why-use-exponential-and-log-in-positional-encoding-of-transformer - formula/articles explanation

        # Initialize the positional encoding matrix 'pe' with zeros.
        # Shape: (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a position tensor containing position indices [0, 1, ..., seq_len - 1].
        # Shape: (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # .unsqueeze(1) or .reshape(seq_len,1)

        # Compute the div_term (divisor term) using the exponential decay function.
        # This term scales the position indices for the sine and cosine functions.
        # Shape: (d_model // 2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *  # Even indices
            (-math.log(10000.0) / d_model)
        )

        # Apply the sine function to even indices in the embeddings.
        # For even dimensions: PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply the cosine function to odd indices in the embeddings.
        # For odd dimensions: PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension to 'pe' for batch compatibility.
        # Shape: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register 'pe' as a buffer to exclude it from model parameters and prevent it from being updated during training.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the positional encoding module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output tensor of the same shape as 'x' with positional encodings added.
        """
        # x has shape (batch_size, seq_len, d_model)
        # Add positional encoding to input embeddings.
        # The positional encoding 'pe' is broadcasted along the batch dimension.
        x = x + self.pe[:, :x.size(1), :]

        # Apply dropout (if any) to prevent overfitting.
        x = self.dropout(x)

        return x


class LayerNormalization(nn.Module):
    """
    Layer Normalization is standard technic in NLP tasks
    LN is used to stabilize the training process and improve the performance of neural networks
    It addresses the internal covariant shift (ICS) problem
    """
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std * self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Feed Forward Block is used in the end of encoder and decoder

    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):
    """

    """
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0  # d_model is not divisible by h

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod  # We don't need to have an instance of this class to use its method. MultiHeadAttention.func_name
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Scale Dot Product Attention is used to calculate the similarity between the words in sequence

        Args:
            :param dropout:
            :param query: given sentence that we focused on (decoder)
            :param key: every sentence to check relationship with Query(encoder)
            :param value: every sentence same with Key (encoder)
            :param mask:
            :return:
        """
        d_k = query.shape[-1]

        # We need transposed key, because we want to compare one query with each key(token embedding)
        # You can imagine it as we are comparing one word to all words in a sequence
        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len,  seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # See the ScaleDotProduct formula
        # http://matrixmultiplication.xyz/ for better understanding of calculation

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)  # it means if mask is true, replace it with -1e9
            # after applying softmax it will be 0
        attention_scores = attention_scores.softmax(dim=1)  # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        """
        Masking is for replacing some words we don't want some words to interact with each other
        We are using this module in decoder and encoder, so basically we use masking in decoder to prevent decoder look
        in the future, while it's generating output sequence, so we are masking the next tokens with the value
        close to -inf, decoder takes into consideration only the previous words
        Args:
            :param q: Query
            :param k: Key
            :param v: Value
            :param mask: Masking is for replacing some words we don't want some words to interact with each other,
            :return:
        """
        query = self.w_q(q)  # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(k)  # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(v)  # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)

        # (Batch, Seq_Len, d_model) --> (Batch, seq_len, heads, d_k) --> (Batch, heads, seq_len, d_k)
        # We want to see full sentence but only smaller parts of embeddings
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  # (Batch, seq_len, heads, d_k) --> (Batch, heads, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Scale dot product to compute similarity
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, heads, seq_len, d_k) --> (Batch, seq_len, heads, d_k) --> (Batch, seq_len, d_model)
        # In order to use view we need to make our tensor .contiguous()
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """

    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """

    """
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    """

    """
    def __init__(self, features: int,  self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(p=dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """

        :param x:
        :param encoder_output:
        :param src_mask: source mask is mask from the decoder block
        :param tgt_mask: target mask is mask from decoder
        :return:
        """
        # decoder self_attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # cross self_attention(query from decoder and key and value from encoder) and mask from the encoder
        x = self.residual_connections[1](x, lambda x: self.self_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    """
    In the original paper "Attention is all you need", on the left side of encoder/decoder there is "Nx", basically
    it means that you can repeat that block many times to improve model performance
    Repeating blocks of encoder/decoder is causes to better performance but also ......
    """
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """

    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask,):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    :param src_vocab_size: source vocab size
    :param tgt_vocab_size: target vocab size
    :param src_seq_len:
    :param tgt_seq_len:
    :param d_model: size of embedding for each token
    :param N: Number of Encoder/Decoder blocks
    :param h: Number of heads, we divide embedding by the number of heads
    :param dropout:
    :param d_ff: Size of hidden layer in the FeedForwardBlock
    :return:
    """

    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block,
                                     dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters with using Xavier form
    for p in transformer.parameters():
        if p.dim() > 1:
             nn.init.xavier_uniform_(p)

    return transformer
