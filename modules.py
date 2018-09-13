# ------------------------
# - Author:  Tao, Tu
# - Date:    2018/8/30
# - Description:
#       Modules used in `Tacotron`.
#
# -----------------------
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparams import Hyperparams as hps


class CharEmbedding(nn.Module):
    """Character embedding
    """

    def __init__(self, input_size, embed_size):
        """Construct CharEmbedding class.
        Args:
            input_size: the number of tokens to embed. E.g., len(vocab).
            embed_size: the dimension of embedding.
        """
        super(CharEmbedding, self).__init__()
        self.net = nn.Embedding(input_size, embed_size,
                                padding_idx=hps.char_set.find('P'))

    def forward(self, x):
        """
        Args:
            x: An index of type torch.LongTensor.
        returns:
            The embedding of `x`.
        """
        return self.net(x)


class Prenet(nn.Module):
    """Prenet of Tacotron
    """

    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(Prenet, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(dropout_rate)),
            ('fc2', nn.Linear(hidden_size, output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(dropout_rate))
        ]))

    def forward(self, x):
        return self.net(x)


class HighwayNet(nn.Module):
    """Highway network
    """

    def __init__(self, input_size, output_size):
        super(HighwayNet, self).__init__()
        self.H = nn.Linear(input_size, output_size)
        self.T = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = F.relu(self.H(x))
        t = torch.sigmoid(self.T(x))
        output = h * t + x * (1. - t)
        return output


class Encoder_CBHG(nn.Module):
    """The encoder CBHG of Tacotron
    """

    def __init__(self, K, input_size, hidden_size):
        super(Encoder_CBHG, self).__init__()
        self.conv_bank = nn.ModuleList()
        for k in range(1, K + 1):
            self.conv_bank.append(Conv1d_SAME(in_channels=input_size,
                                              out_channels=hidden_size,
                                              kernel_size=k))
        self.bn_bank = nn.BatchNorm1d(hidden_size * K)
        self.max_pool = MaxPool1d_SAME(kernel_size=2)
        self.conv_proj_1 = Conv1d_SAME(in_channels=hidden_size * K,
                                       out_channels=hidden_size,
                                       kernel_size=3)
        self.bn_proj_1 = nn.BatchNorm1d(hidden_size)
        self.conv_proj_2 = Conv1d_SAME(in_channels=hidden_size,
                                       out_channels=hidden_size,
                                       kernel_size=3)
        self.bn_proj_2 = nn.BatchNorm1d(hidden_size)
        self.highway_list = nn.ModuleList()
        for k in range(hps.num_highway):
            self.highway_list.append(HighwayNet(input_size=hidden_size,
                                                output_size=hidden_size))
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          bidirectional=True,
                          batch_first=True)
    
    def forward(self, x, gru_hidden=None):
        """
        Args:
            x: A tensor with shape (batch_size, seq_length, channels)
        Returns:
            A tensor with shape (batch_size, seq_length, 2*hidden_size)
        """
        x = x.transpose(1, 2)  # Shape: (batch_size, channels, seq_length)
        bank_out = []
        # Conv1d bank
        for k, conv in enumerate(self.conv_bank):
            output = conv(x)  # Shape: (batch_size, hidden_size, seq_length)
            bank_out.append(output)
        # Shape: (batch_size, K * hidden_size, seq_length)
        y = torch.cat(bank_out, dim=1)
        y = F.relu(self.bn_bank(y))
        # MaxPool1d
        y = self.max_pool(y)
        # Projection 1
        # Shape: (batch_size, hidden_size, seq_length)
        y = F.relu(self.bn_proj_1(self.conv_proj_1(y)))
        # Projection 2
        # Shape: (batch_size, hidden_size, seq_length)
        y = self.bn_proj_2(self.conv_proj_2(y))
        # Residual connection
        y = y + x
        # Highway network
        y = y.transpose(1, 2)  # Shape: (batch_size, seq_length, channels)
        for highway in self.highway_list:
            y = highway(y)
        # GRU
        self.gru.flatten_parameters()
        if gru_hidden is not None:
            out, state = self.gru(y, gru_hidden)
        else:
            out, state = self.gru(y)
        return out, state


class Decoder_CBHG(nn.Module):
    """The decoder CBHG of Tacotron
    """

    def __init__(self, K, input_size, hidden_size):
        super(Decoder_CBHG, self).__init__()
        self.conv_bank = nn.ModuleList()
        for k in range(1, K + 1):
            self.conv_bank.append(Conv1d_SAME(in_channels=input_size,
                                              out_channels=hidden_size,
                                              kernel_size=k))
        self.bn_bank = nn.BatchNorm1d(hidden_size * K)
        self.max_pool = MaxPool1d_SAME(kernel_size=2)
        self.conv_proj_1 = Conv1d_SAME(in_channels=hidden_size * K,
                                       out_channels=256,
                                       kernel_size=3)
        self.bn_proj_1 = nn.BatchNorm1d(256)
        self.conv_proj_2 = Conv1d_SAME(in_channels=256,
                                       out_channels=hps.n_mels,
                                       kernel_size=3)
        self.bn_proj_2 = nn.BatchNorm1d(hps.n_mels)
        self.highway_proj = nn.Linear(hps.n_mels, hidden_size)
        self.highway_list = nn.ModuleList()
        for k in range(hps.num_highway):
            self.highway_list.append(HighwayNet(input_size=hidden_size,
                                                output_size=hidden_size))
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          bidirectional=True,
                          batch_first=True)

    def forward(self, x, gru_hidden=None):
        """
        Args:
            x: A tensor with shape (batch_size, seq_length, channels)
        Returns:
            A tensor with shape (batch_size, seq_length, 2*hidden_size)
        """
        x = x.transpose(1, 2)  # Shape: (batch_size, channels, seq_length)
        bank_out = []
        # Conv1d bank
        for k, conv in enumerate(self.conv_bank):
            output = conv(x)  # Shape: (batch_size, hidden_size, seq_length)
            bank_out.append(output)
        # Shape: (batch_size, K * hidden_size, seq_length)
        y = torch.cat(bank_out, dim=1)
        y = F.relu(self.bn_bank(y))
        # MaxPool1d
        y = self.max_pool(y)
        # Projection 1
        # Shape: (batch_size, 256, seq_length)
        y = F.relu(self.bn_proj_1(self.conv_proj_1(y)))
        # Projection 2
        # Shape: (batch_size, 80, seq_length)
        y = self.bn_proj_2(self.conv_proj_2(y))
        # Residual connection
        #y = y + x
        # Highway network
        y = y.transpose(1, 2)  # Shape: (batch_size, seq_length, 80)
        # Shape: (batch_size, seq_length, hidden_size)
        y = self.highway_proj(y)
        for highway in self.highway_list:
            y = highway(y)
        # GRU
        self.gru.flatten_parameters()
        if gru_hidden is not None:
            out, state = self.gru(y, gru_hidden)
        else:
            out, state = self.gru(y)
        return out, state


class Attention(nn.Module):
    """Implement Bahdanau attention mechanism.
    """

    def __init__(self, query_size, context_size, hidden_size=None):
        super(Attention, self).__init__()
        if hidden_size is None:
            hidden_size = context_size
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_c = nn.Linear(context_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.normal(mean=torch.zeros(
            hidden_size), std=torch.ones(hidden_size)))

    def forward(self, query, context):
        """
        Args:
            query: A tensor with shape (batch_size, 1, query_size)
            context: A tensor with shape (batch_size, seq_length, context_size)
        Returns:
            The alignment tensor with shape (batch, seq_length)
        """
        batch_size = context.size(0)
        seq_len = context.size(1)
        # Shape: (batch_size, seq_length, query_size)
        query_tiled = query.repeat(1, seq_len, 1)
        # Shape: (batch_size, seq_length, hidden_size)
        info_matrix = torch.tanh(self.W_q(query_tiled) + self.W_c(context))
        # Shape: (batch_size, hidden_size, 1)
        v_tiled = self.v.unsqueeze(0).repeat(batch_size, 1).unsqueeze(2)  
        # Shape: (batch_size, seq_length)
        energy = torch.bmm(info_matrix, v_tiled).squeeze(2)  
        alignment = F.softmax(energy, dim=1)
        return alignment


class AttentionRNN(nn.Module):
    """Attention RNN in original Tacotron paper.
    """

    def __init__(self, input_size, hidden_size, text_embed_size):
        super(AttentionRNN, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size, batch_first=True)
        self.text_embed_size = text_embed_size
        self.attn = Attention(query_size=hidden_size, 
                              context_size=text_embed_size)

    def forward(self, x, memory, gru_hidden=None):
        """
        Args:
            x: A tensor with shape (batch_size, 1, input_size)
            memory: the output of `Encoder` with shape (batch_size, seq_length, text_embed_size).
        Returns:
            out: the output of gru with shape (batch_size, 1, hidden_size + text_embed_size).
            a: attention weight with shape (batch_size, seq_length).
        """

        self.gru.flatten_parameters()
        # Shape: (batch_size, 1, hidden_size)
        if gru_hidden is not None:
            h, state = self.gru(x, gru_hidden)
        else:
            h, state = self.gru(x)
        # Shape: (batch_size, seq_length)
        a = self.attn(query=h, context=memory)
        # Shape: (batch_size, seq_length, text_embed_size)
        a_tile = a.unsqueeze(2).repeat(1, 1, self.text_embed_size)
        # Shape: (batch_size, text_embed_size)
        context = torch.sum(a_tile * memory, dim=1)
        # Shape: (batch_size, 1, text_embed_size + hidden_size)
        out = torch.cat([context.unsqueeze(1), h], dim=2)
        return out, a, state


class DecoderRNN(nn.Module):
    """Decoder RNN in original Tacotron paper.
    """

    def __init__(self, input_size, output_size, r=2):
        """
        Args:
            r: An int, reduction factor.
        """
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.r = r
        self.gru_1 = nn.GRU(input_size=input_size,
                            hidden_size=input_size, batch_first=True)
        self.gru_2 = nn.GRU(input_size=input_size,
                            hidden_size=input_size, batch_first=True)
        self.fc = nn.Linear(input_size, r * output_size)

    def forward(self, x, gru_hidden_1=None, gru_hidden_2=None):
        """
        Args:
            x: A tensor with shape (batch_size, seq_len=1, input_size)
        """
        # Shape: (batch_size, 1, input_size)
        self.gru_1.flatten_parameters()
        if gru_hidden_1 is not None:
            dec_1, state_1 = self.gru_1(x, gru_hidden_1)
        else:
            dec_1, state_1 = self.gru_1(x)  
        # Shape: (batch_size, 1, input_size)
        self.gru_2.flatten_parameters()
        if gru_hidden_2 is not None:
            dec_2, state_2 = self.gru_2(dec_1 + x, gru_hidden_2)
        else:
            dec_2, state_2 = self.gru_2(dec_1 + x)
        # Shape: (batch_size, 1, r*output_size)
        out = self.fc(dec_1 + dec_2 + x)
        out = out.squeeze(1).view(x.size(0), self.r, self.output_size)
        return out, state_1, state_2


def _padding(kernel_size):
    return int(kernel_size // 2)


def _adjust_conv_dim(x, kernel_size):
    # discard the last one if kernel_size % 2 == 0
    return x[:, :, :-1] if kernel_size % 2 == 0 else x


class Conv1d_SAME(nn.Module):
    """For padding with `SAME` mode
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Conv1d_SAME, self).__init__()
        self.kernel_size = kernel_size
        self.net = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=_padding(kernel_size), bias=bias)

    def forward(self, x):
        return _adjust_conv_dim(self.net(x), self.kernel_size)


class MaxPool1d_SAME(nn.Module):
    """For padding with `SAME` mode
    """

    def __init__(self, kernel_size):
        super(MaxPool1d_SAME, self).__init__()
        self.kernel_size = kernel_size
        self.net = nn.MaxPool1d(kernel_size=kernel_size,
                                stride=1, padding=_padding(kernel_size))

    def forward(self, x):
        return _adjust_conv_dim(self.net(x), self.kernel_size)
