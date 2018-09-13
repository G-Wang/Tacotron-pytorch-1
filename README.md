# A-Pytorch-Implementation-of-Tacotron-End-to-end-Text-to-speech-Deep-Learning-Model

## Samples
See the samples at *samples/* which are generated after training 200k.


## Usage

* Train
```bash
# If you have pretrain model, add --ckpt <ckpt_path>
$ python main.py --train --cuda
```

* Evaluation
```bash
$ python main.py --eval --cuda --ckpt <ckpt_timestep.pth.tar>
```
