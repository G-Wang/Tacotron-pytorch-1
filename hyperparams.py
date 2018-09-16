# ------------------------
# - Author:  Tao, Tu
# - Date:    2018/8/30
# - Description:
#       The hyperparameters.
#
# -----------------------
class Hyperparams(object):
    # Test
    eval_texts = ['Welcome to National Taiwan University speech lab.',
                  'The birch canoe slid on the smooth planks.',
                  'Glue the sheet to the dark blue background.',
                  'It\'s easy to tell the depth of a well.', 
                  'These days a chicken leg is a rare dish.',
                  'Rice is often served in round bowls.',
                  'The juice of lemons makes fine punch.',
                  'The box was thrown beside the parked truck.',
                  'The hogs were fed chopped corn and garbage.',
                  'Four hours of steady work faced us.',
                  'Large size in stockings is hard to sell.',
                  'The boy was there when the sun rose.',
                  'A rod is used to catch pink salmon.',
                  'The source of the huge river is the clear spring.',
                  'Kick the ball straight and follow through.',
                  'Help the woman get back to her feet.',
                  'A pot of tea helps to pass the evening.',
                  'Smoky fires lack flame and heat.',
                  'The soft cushion broke the man\'s fall.',
                  'The salt breeze came across from the sea.',
                  'The girl at the booth sold fifty bonds.'
                 ]

    # Path
    meta_path = '/home/ttao/Datasets/LJSpeech/LJSpeech-1.1/metadata.csv'
    wav_dir = '/home/ttao/Datasets/LJSpeech/LJSpeech-1.1/wavs'

    # Save
    save_model_every_step = 1000 
    save_result_every_step = 1000

    # Log
    log_every_step = 5

    # Dataset
    char_set = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P for padding, E for eos
    text_min_length = 0
    text_max_length = 200
    bucket = True

    # Network configuration
    lr = 1e-3
    warmup_step = 4000
    batch_size = 8 
    reduction_factor = 5
    prenet_size = [256, 128]
    prenet_dropout_rate = 0.5
    embed_size = 256
    num_highway = 4
    K_encoder = 16
    K_decoder = 8
    clip_norm = 5.0

    # Inference configuration
    max_infer_step = 500 // reduction_factor

    # Signal processing
    sampling_rate = 22050
    pre_emphasis = 0.97
    n_fft = 2048
    n_mels = 80
    frame_hop = 0.0125  # unit: sec, i.e., 12.5 ms
    frame_len = 0.05    # unit: sec
    hop_length = int(sampling_rate * frame_hop)  # unit: sample
    win_length = int(sampling_rate * frame_len)  # unit: sample

    ref_db = 20
    max_db = 100
    GL_n_iter = 200  # Griffin-Lim inverse iteration
