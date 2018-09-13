# ------------------------
# - Author:  Tao, Tu
# - Date:    2018/8/30
# - Description:
#       Group modules to form networks.
#
# -----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparams import Hyperparams as hps
from modules import CharEmbedding, Prenet, Encoder_CBHG, Decoder_CBHG, AttentionRNN, DecoderRNN, Attention


class Tacotron(nn.Module):
    """Tacotron
    """

    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(
            vocab_size=len(hps.char_set),
            embed_size=hps.embed_size)
        self.decoder_mel = Decoder_Mel(
            input_size=hps.n_mels,
            hidden_size=hps.embed_size,
            text_embed_size=hps.embed_size,
            reduction_factor=hps.reduction_factor)
        self.decoder_mag = Decoder_Mag(
            input_size=hps.n_mels,
            hidden_size=hps.embed_size // 2)

    def forward(self, text, frames):
        """
        Args:
            text: a batch of index sequence with shape (batch_size, text_seq_lenth)
            frames: 
                if training: 
                    the input frames of decoder prenet. 
                    Shape: (batch_size, frame_seq_length, hps.n_mels)
                if testing : 
                    a GO frame. Shape: (1, 1, hps.n_mels)
        Returns:
            The magnitude spectrogram with shape (batch_size, T, F)
        """

        text_embed = self.encoder(text)
        mel_pred, a, state_attn, state_dec_1, state_dec_2 = self.decoder_mel(frames[:, 0, :].unsqueeze(1), text_embed)
        # Shape: (batch_size, 1, max_time_step)
        attn = a.unsqueeze(1)
        if self.training:
            time_step = frames.size(1)
            for t in range(1, time_step):
                # Shape: (batch_size, r, hps.n_mels)
                pred, a, state_attn, state_dec_1, state_dec_2 = self.decoder_mel(frames[:, t, :].unsqueeze(1), text_embed, state_attn, state_dec_1, state_dec_2) 
                attn = torch.cat([attn, a.unsqueeze(1)], dim=1) 
                mel_pred = torch.cat([mel_pred, pred], dim=1)
            mag_pred = self.decoder_mag(mel_pred)
        else:
            for t in range(hps.max_infer_step):
                # Shape: (1, r, hps.n_mels)
                pred, a, state_attn, state_dec_1, state_dec_2 = self.decoder_mel(mel_pred[:, -1, :].unsqueeze(1), text_embed, state_attn, state_dec_1, state_dec_2)
                attn = torch.cat([attn, a.unsqueeze(1)], dim=1) 
                mel_pred = torch.cat([mel_pred, pred], dim=1)
            mag_pred = self.decoder_mag(mel_pred)
        return {'mel': mel_pred, 'mag': mag_pred, 'attn': attn}


class Encoder(nn.Module):
    """Character embedding layer + pre-net + CBHG
    """

    def __init__(self, vocab_size, embed_size):
        super(Encoder, self).__init__()
        self.embed = CharEmbedding(vocab_size, embed_size)
        self.prenet = Prenet(
            input_size=embed_size,
            hidden_size=hps.prenet_size[0],
            output_size=embed_size // 2,
            dropout_rate=hps.prenet_dropout_rate)
        self.CBHG = Encoder_CBHG(
            K=hps.K_encoder,
            input_size=embed_size // 2,
            hidden_size=embed_size // 2)

    def forward(self, x):
        """
        Args:
            x: index sequence with shape (batch_size, seq_length).
        Returns:
            A tensor with shape (batch_size, seq_length, embed_size).
        """
        y = self.embed(x)
        y = self.prenet(y)
        out, _ = self.CBHG(y)
        return out


class Decoder_Mel(nn.Module):
    """Pre-net + attention RNN + Decoder RNN. Decode `reduction_factor` mel-vectors a time.
    """

    def __init__(self, input_size, hidden_size, text_embed_size, reduction_factor=2):
        super(Decoder_Mel, self).__init__()
        self.text_embed_size = text_embed_size
        self.prenet = Prenet(
            input_size=input_size,
            hidden_size=hps.prenet_size[0],
            output_size=hidden_size // 2,
            dropout_rate=hps.prenet_dropout_rate)
        self.attnRNN = AttentionRNN(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            text_embed_size=text_embed_size)
        self.attn = Attention(
            query_size=hidden_size,
            context_size=text_embed_size)
        self.decRNN = DecoderRNN(
            input_size=hidden_size + text_embed_size,
            output_size=hps.n_mels,
            r=reduction_factor)

    def forward(self, frame, memory, gru_hidden_attn=None, gru_hidden_dec_1=None, gru_hidden_dec_2=None):
        """
        Args:
            frame: a frame with shape (batch_size, 1, input_size).
            memory: the output of `Encoder` with shape (batch_size, seq_length, text_embed_size).
        Returns:
            A tensor with shape (batch_size, reduction_factor, hps.n_mels)
        """
        h = self.prenet(frame)
        h, a, state_attn = self.attnRNN(h, memory, gru_hidden_attn)
        out, state_dec_1, state_dec_2 = self.decRNN(h, gru_hidden_dec_1, gru_hidden_dec_2)
        return out, a, state_attn, state_dec_1, state_dec_2


class Decoder_Mag(nn.Module):
    """CBHG + projection layer. Decode the magnitude spectrogram given mel-spectrogram.
    """

    def __init__(self, input_size, hidden_size):
        super(Decoder_Mag, self).__init__()
        self.CBHG = Decoder_CBHG(
            K=hps.K_decoder,
            input_size=input_size,
            hidden_size=hidden_size)
        self.proj = nn.Linear(2 * hidden_size, 1 + hps.n_fft // 2)

    def forward(self, x):
        """
        Args:
            x: a tensor with shape (batch_size, audio_seq_length, input_size)
        Returns:
            A tensor with shape (batch_size, audio_seq_length, 1 + n_fft//2)
        """
        y, _ = self.CBHG(x)
        out = self.proj(y)
        return out
