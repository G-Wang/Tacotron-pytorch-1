# ------------------------
# - Author:  Tao, Tu
# - Date:    2018/8/30
# - Description:
#       Train or evaluate the model.
#
# -----------------------
import os
import sys
import time
import random
import argparse
import numpy as np
from torch import nn, optim
from dataset import LJSpeech_Dataset, collate_fn
from network import Tacotron
from hyperparams import Hyperparams as hps
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils import save_alignment, save_spectrogram, mag2wav, save_wav, text_normalize, sent2idx

#torch.backends.cudnn.benchmark = True

_to_save = ['model', 'optimizer', 'step', 'epoch']

def save(filepath, **ckpt):
    """Save current training status.
    Args:
        filepath: where to save the checkpoint.
        ckpt: checkpoint that includes those in `_to_save`.
    """
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    for key in _to_save:
        if key not in ckpt:
            raise Exception('{} need to be saved.'.format(key))

    torch.save(ckpt, filepath)


def load(filepath):
    """Load the checkpoint.
    Args:
        filepath: the checkpoint path.
    Returns:
        The checkpoint file includeing those in `_to_save`.
    """
    ckpt = torch.load(filepath)
    return ckpt


def train(model, loader, optimizer, criterion, scheduler, step, epoch, device, args):
    before_load = time.time()
    # Start training
    model.train()
    while True:
        for batch in loader:
            # torch.LongTensor, (batch_size, seq_length)
            txt = batch['text']
            # torch.Tensor, (batch_size, max_time, hps.n_mels)
            mel = batch['mel']
            # torch.Tensor, (batch_size, max_time, hps.n_fft)
            mag = batch['mag']
            # torch.LongTensor, (batch_size, )
            txt_len = batch['text_length']
            frame_len = batch['frame_length']

            if hps.bucket:
                # If bucketing, the shape will be (1, batch_size, ...)
                txt = txt.squeeze(0)
                mel = mel.squeeze(0)
                mag = mag.squeeze(0)
                txt_len = txt_len.squeeze(0)
                frame_len = frame_len.squeeze(0)
            # GO frame
            GO_frame = torch.zeros(mel[:, :1, :].size())
            if args.cuda:
                txt = txt.to(device)
                mel = mel.to(device)
                mag = mag.to(device)
                GO_frame = GO_frame.to(device)

            # Model prediction
            decoder_input = torch.cat([GO_frame, mel[:, hps.reduction_factor::hps.reduction_factor, :]], dim=1)

            load_time = time.time() - before_load
            before_step = time.time()

            _batch = model(text=txt, frames=decoder_input, text_length=txt_len, frame_length=frame_len)
            _mel = _batch['mel']
            _mag = _batch['mag']
            _attn = _batch['attn']

            # Optimization
            optimizer.zero_grad()
            loss_mel = criterion(_mel, mel)
            loss_mag = criterion(_mag, mag)
            loss = loss_mel + loss_mag
            loss.backward()
            # Gradient clipping
            total_norm = clip_grad_norm_(model.parameters(), max_norm=hps.clip_norm)
            # Apply gradient
            optimizer.step()
            # Adjust learning rate
            scheduler.step()
            process_time = time.time() - before_step
            if step % hps.log_every_step == 0:
                lr_curr = optimizer.param_groups[0]['lr']
                log = '[{}-{}] total_loss: {:.3f}, mel_loss: {:.3f}, mag_loss: {:.3f}, grad: {:.3f}, lr: {:.3e}, time: {:.2f} + {:.2f} sec'.format(epoch, step, loss.item(), loss_mel.item(), loss_mag.item(), total_norm, lr_curr, load_time, process_time)
                print(log)
            if step % hps.save_model_every_step == 0:
                save(filepath='tmp/ckpt/ckpt_{}.pth.tar'.format(step),
                     model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     step=step,
                     epoch=epoch)

            if step % hps.save_result_every_step == 0:
                sample_idx = random.randint(0, hps.batch_size-1)
                attn_sample = _attn[sample_idx].detach().cpu().numpy()
                mag_sample = _mag[sample_idx].detach().cpu().numpy()
                wav_sample = mag2wav(mag_sample)
                # Save results
                save_alignment(attn_sample, step, 'tmp/plots/attn_{}.png'.format(step))
                save_spectrogram(mag_sample, 'tmp/plots/spectrogram_{}.png'.format(step))
                save_wav(wav_sample, 'tmp/results/wav_{}.wav'.format(step))
            before_load = time.time()
            step += 1
        epoch += 1

def evaluation(model, step, device, args):
    # Evaluation
    model.eval()
    with torch.no_grad():
	    # Preprocessing eval texts
        print('Start generating evaluation speeches...')
        n_eval = len(hps.eval_texts)
        for i in range(n_eval):
            sys.stdout.write('\rProgress: {}/{}'.format(i+1, n_eval))
            sys.stdout.flush()
            text = hps.eval_texts[i]
            text = text_normalize(text)
            
            txt_id = sent2idx(text) + [hps.vocab.find('E')]
            txt_len = len(txt_id)
            GO_frame = torch.zeros(1, 1, hps.n_mels)

            # Shape: (1, seq_length)
            txt = torch.LongTensor([txt_id])
            txt_len = torch.LongTensor([txt_len])
            if args.cuda:
                GO_frame = GO_frame.cuda()
                txt = txt.cuda()
                txt_len.cuda()
            _batch = model(text=txt, frames=GO_frame, text_length=txt_len)
            mel = _batch['mel'][0]
            mag = _batch['mag'][0]
            attn = _batch['attn'][0]
            if args.cuda:
                mel = mel.cpu()
                mag = mag.cpu()
                attn = attn.cpu()
            mel = mel.numpy()
            mag = mag.numpy()
            attn = attn.numpy()

            wav = mag2wav(mag)
            save_alignment(attn, step, 'eval/plots/attn_{}.png'.format(text))
            save_spectrogram(mag, 'eval/plots/spectrogram_[{}].png'.format(text))
            save_wav(wav, 'eval/results/wav_{}.wav'.format(text))
        sys.stdout.write('\n')


def run(args):
    # Check cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Data
    if hps.bucket:
        dataset = LJSpeech_Dataset(meta_file=hps.meta_path, wav_dir=hps.wav_dir, batch_size=hps.batch_size, do_bucket=True)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4)
    else:
        dataset = LJSpeech_Dataset(meta_file=hps.meta_path, wav_dir=hps.wav_dir)
        loader = DataLoader(
            dataset,
            batch_size=hps.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn)

    # Network
    model = Tacotron()
    criterion = nn.L1Loss()
    if args.cuda:
        model = nn.DataParallel(model.to(device))
        criterion = criterion.to(device)
    # The learning rate scheduling mechanism in "Attention is all you need"
    lr_lambda = lambda step: hps.warmup_step ** 0.5 * min((step+1) * (hps.warmup_step ** -1.5), (step+1) ** -0.5)
    optimizer = optim.Adam(model.parameters(), lr=hps.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 1
    epoch = 1
    # Load model
    if args.ckpt:
        ckpt = load(args.ckpt)
        step = ckpt['step']
        epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
            last_epoch=step)

    if args.eval:
        evaluation(model, step, device, args)

    if args.train:
        train(model, loader, optimizer, criterion, scheduler, step, epoch, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tacotron Configuration')
    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--eval', action='store_true', help='eval mode')
    parser.add_argument('--cuda', action='store_true', help='use gpu')
    parser.add_argument('--ckpt', default=None, type=str, help='e.g., "ckpt/model_{}.pth.tar"')
    args = parser.parse_args()
    run(args)

