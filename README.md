# tensorflow-textgen
This is char recurrent nueral network which tries to learn to generate character by character a text similar to the one on which is trained on.

This project was initially developed as part of the Intelligent System and Pattern Recognition course's exam @ University of Pisa.

The code is based on the [Tensorflow tutorial](https://www.tensorflow.org/text/tutorials/text_generation) on text generation.

## usage
I developed and tested the network using the official Tensorflow with GPU docker environment ([link](https://hub.docker.com/r/tensorflow/tensorflow/)) with **tensorflow 2.5**. 
In order to run the scripts, download the repo:
```
git clone https://github.com/giacomofrigo/tensorflow-textgen
cd tensorflow-textgen
```
and start docker container
```
docker run --gpus=all -it --rm -v /$PWD:/tmp -w /tmp tensorflow/tensorflow:latest-gpu
```

If you wnat to use also tensorboard remember to expose port 6006:

```
docker run --gpus=all -it --rm -v /$PWD:/tmp -w /tmp -p 6006:6006 tensorflow/tensorflow:latest-gpu
```
## training
To train the model run `python train.py`. Run ` python train.py --help` to see the list of possible paramters and their default values.
```
root@23833e6eae4f:/tmp# python train.py --help
usage: train.py [-h] [--save_dir SAVE_DIR] [--log_dir LOG_DIR]
                [--validation_file VALIDATION_FILE]
                [--embedding_dim EMBEDDING_DIM] [--rnn_units RNN_UNITS]
                [--num_layers NUM_LAYERS] [--seq_length SEQ_LENGTH]
                [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                [--learning_rate LEARNING_RATE] [--dropout DROPOUT]
                input_file

positional arguments:
  input_file

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   directory to store checkpointed models and model
                        configuration (default: save)
  --log_dir LOG_DIR     directory to store tensorflow logs, that can be used
                        for tensorboard (default: {save_dir}/logs)
  --validation_file VALIDATION_FILE
                        validation dataset. if passed the model is tested on
                        the validation dataset at the end of each epoch
                        (default: None)
  --embedding_dim EMBEDDING_DIM
                        Dimension of the embedding layer which is the input
                        layer. A trainable lookup table that will map each
                        character-ID to a vector with args.embedding_dim
                        dimensions (default: 128)
  --rnn_units RNN_UNITS
                        size of RNN hidden state (default: 256)
  --num_layers NUM_LAYERS
                        number of layers in the RNN (default: 1)
  --seq_length SEQ_LENGTH
                        RNN sequence length. Number of timesteps to unroll
                        for. (default: 100)
  --batch_size BATCH_SIZE
                        minibatch size. Number of sequences propagated through
                        the network in parallel. Pick batch-sizes to fully
                        leverage the GPU (e.g. until the memory is filled up)
                        commonly in the range 10-500. (default: 64)
  --num_epochs NUM_EPOCHS
                        number of epochs. Number of full passes through the
                        training examples. (default: 50)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.002)
  --dropout DROPOUT     probability of keeping weights in the hidden layer
                        (default: 0.1)


```

During training at the end of each epoch the model weights are saved to the specified `save_dir`. The sample script will then use the last checkpoint in order to sample new text.
The script automatically saves tensorflow logs allowing tensorboard visualization. The script also tests the model by generating some characters at the end of each epoch. 
The complete history of generated texts will be then exported to `result.json` at the end of the training process.

## sample
Sample from the trained model is quite straight forward, you only need to run `python sample.py` and passing to it the `save_dir` of the trainend model.
The other parameters of the sample script is reported below.

```
root@f95c3dfb368c:/tmp# python sample.py --help
usage: sample.py [-h] [-n N] [--prime PRIME] [--temperature TEMPERATURE]
                 save_dir

positional arguments:
  save_dir              checkpoints and configurations directory

optional arguments:
  -h, --help            show this help message and exit
  -n N                  number of characters to sample (default: 500)
  --prime PRIME         prime text (default: :)
  --temperature TEMPERATURE
                        sampling temperature (default: 1)
                        
```

## tensorboard 
In order to run tensorboard, from inside the docker container run:
```
tensorboard --logdir %save_dir% --bind_all
```
Passing as `%save_dir$` the save directory of the trained model. 




