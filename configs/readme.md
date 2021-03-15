### Model related arguments:
* id - a name for identifying the model
* num_mix - number of sounds to mix
* arch_sound - architecture of net_sound
* arch_frame - architecture of net_frame
* arch_synthesizer - architecture of net_synthesizer
* num_channels - number of channels
* num_frames - number of frames
* stride_frames - sampling stride of frames
* img_pool - avg or max pool image features
* img_activation - activation on the image features
* sound_activation - activation on the sound features
* output_activation - activation on the output
* binary_mask - whether to use binary masks
* mask_thres - threshold in the case of binary masks
* loss - loss function to use
* weighted_loss - weighted_loss
* log_freq - log frequency scale

### Data related arguments
* num_gpus - number of gpus to use
* batch_size_per_gpu - input batch size
* workers - number of data loading workers
* num_val - number of images to evaluate
* num_vis - number of images to display during evaluation
* aud_len - sound length
* aud_rate - sound sampling rate
* stft_frame - stft frame length
* stft_hop - stft hop length
* img_size - size of input frame
* frame_rate - video frame sampling rate

### Misc arguments
* seed - manual random seed
* ckpt - folder to output checkpoints
* vis_dir - folder to output visulaisation (relative to ckpt)
* disp_iter - frequency to display
* eval_epoch - frequency to evaluate

### Train arguments
* list_train - path to train.csv
* list_val - path to train.val
* dup_trainset - duplicate so that one epoch has more iters
* num_epoch - epoch_to_train_for
* lr_frame - learning rate for frame network
* lr_sound - learning rate for audio network
* lr_synthesizer - learning rate for synthesizer network
* lr_steps - steps to drop LR in epochs
* beta1 - momentum for sgd, beta1 for adam
* weight_decay - weights regularizer

* continue_training - continue train model from last epoch
* finetune - freeze frame network and audio network, path to model
* checkpoint_epoch - save model checkpoint every _ epoch
