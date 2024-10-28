from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    """

    """
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]') or 0], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]') or 0], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]') or 0], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Model works only with fixed seq_len, if there are not enough words to fill seq_len fully then
        # we replace the rest of positions with padding tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # [SOS] nad [EOS] tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # only [SOS] special token

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Add SOS and EOS to the source text (encoder input)
        # 1. Start with SOS token
        # 2. Add the encoded input tokens
        # 3. Add EOS token
        # 4. Pad with PAD tokens to reach seq_len
        # This forms the complete input sequence for the encoder
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to the target text (decoder input)
        # The decoder input is used during training and inference:
        # 1. It's shifted one position to the right, starting with SOS token
        # 2. During training, it provides the correct previous token for the model to predict the next one
        # 3. At inference time, it's used to generate the translation step by step
        # 
        # The main differences between decoder_input and label are:
        # - decoder_input starts with SOS and doesn't include EOS
        # - label includes the full target sequence with EOS, but no SOS
        # - decoder_input is used as input to the decoder, while label is used to compute the loss
        # - During training, decoder_input helps the model learn to predict the next token given the previous ones
        # - At inference, the model uses its own predictions as decoder_input to generate the translation
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Label is used for training the model, while decoder_input is used during inference
        # The main differences are:
        # 1. Label includes the full target sequence with EOS token, used to compute loss
        # 2. Decoder_input is shifted one position to the right, starting with SOS token
        # 3. During training, the model learns to predict the next token given previous ones
        # 4. At inference time, the model uses its own predictions as decoder_input
        # Add EOS to the label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)

            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)

            # We need mask to prevent padding tokens from being used in self-attention calculation
            # We use unsqueeze(0) twice to add two dimensions:
            # 1. First unsqueeze(0): Adds a batch dimension
            #    Example: [seq_len] -> [1, seq_len]
            # 2. Second unsqueeze(0): Adds a head dimension for multi-head attention
            #    Example: [1, seq_len] -> [1, 1, seq_len]
            # This results in a mask shape compatible with attention operations: [batch_size, num_heads, seq_len]
            # Example:
            #   Input:  [0, 1, 1, 0] (where 1 is non-pad, 0 is pad)
            #   Output: [[[0, 1, 1, 0]]] (3D tensor with shape [1, 1, 4])

            # Explanation: We use 'encoder_input != self.pad_token' because:
            # 1. It creates a boolean mask where True (1) represents actual input tokens
            #    and False (0) represents padding tokens.
            # 2. This mask is crucial for the self-attention mechanism to ignore padding tokens
            #    when computing attention weights, ensuring that the model focuses only on
            #    relevant input information.
            # 3. By comparing with self.pad_token, we accurately identify which positions
            #    in the input sequence are padding and which are actual input tokens.
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),

            # Explanation of the decoder_mask:
            # 1. .int(): Converts the boolean mask to integers (0 and 1).
            # 2. & causal_mask(decoder_input.size(0)): Applies a causal mask to ensure that the model
            #    can only attend to previous tokens in the sequence during training.
            #
            # The resulting mask combines padding information and causal constraints:
            # - It prevents attention to padding tokens.
            # - It enforces causality, ensuring each position can only attend to itself and previous positions.
            #
            # Shape explanation:
            # (1, 1, seq_len) & (1, seq_len, seq_len) -> (1, seq_len, seq_len)
            # The broadcasting rules expand the first tensor to match the second tensor's shape.

            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt": tgt_text
        }
    

def causal_mask(size):
    """
    Generate a causal mask for a given sequence size.

    This function creates a lower triangular mask that prevents the model
    from attending to future tokens during training or inference.

    Args:
        size (int): The size of the sequence for which to generate the mask.

    Returns:
        torch.Tensor: A boolean tensor of shape (1, size, size) where True values
        allow attention and False values prevent it. The upper triangle (including
        the diagonal) contains False values, creating the causal property.

    Example:
        >>> causal_mask(3)
        tensor([[[ True, False, False],
                 [ True,  True, False],
                 [ True,  True,  True]]])

    Note:
        This mask is crucial for autoregressive models to maintain causality
        by ensuring that each position can only attend to itself and previous positions. 
        https://pytorch.org/docs/stable/generated/torch.triu.html - link with explanation
        torch.triu() - returns the matrix with zeroed values along diagonale
    """
    # Create an upper triangular matrix with 1s above the diagonal
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    # Invert the mask: 1s become 0s and 0s become 1s, then convert to boolean
    return mask == 0