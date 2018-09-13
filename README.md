# A Pytorch Implementation of Tacotron: End-to-end Text-to-speech Deep-Learning Model

## Samples
The sample texts is based on [Harvard Sentences](http://www.cs.columbia.edu/~hgs/audio/harvard.html). See the samples at `samples/` which are generated after training 200k.

## Alignment
![alignment](alignment.gif)


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



## Refenrence
1. (Tensorflow) Kyubyong's [implementatino](https://github.com/Kyubyong/tacotron)
2. (Tensorflow) acetylSv's [implementation](https://github.com/acetylSv/GST-tacotron)
3. (Pytorch)    soobinseo's [implementaition](https://github.com/soobinseo/Tacotron-pytorch)  

Finally, I have to say this work is highly based on Kyubyong's work, so you are a tensorflower, you may want to see his work.
