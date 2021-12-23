# Attention Based Grapheme To Phoneme
The G2P algorithm is used to generate the most probable pronunciation for a word not contained in the lexicon dictionary.


### Test
```python
import torch
model = torch.hub.load(repo_or_dir='kompotiks/G2P', 
                       model='g2p_ru_2', 
                       force_reload=True)
print(model('привет'))
```
To get pronunciation of a word:
```
python -m g2p --word привет
p.r.i.v.jO.t
```


## Dataset
Currently the following languages are supported:
1. RU: Russian

You could easily provide and use your own language specific pronunciatin doctionary for training G2P.
More details about data preparation and contribution could be found in ```resources```.<br/>
Feel free to provide resources for other languages.



## Attention Model
Both encoder-decoder seq2seq model and attention model could handle G2P problem.
Here we train attention based model.
![attention model](attention/attention-bidi.jpg)
The encoder model get sequence of graphemes and produces states at each timestep.
Encoder states used during attention decoding.
The decoder attends to appropriate encoder state (according to its state) and produces phonemes.


### Train
To start training the model run:
```
python train.py
```
You can also use tensorboard to check the training loss:
```
tensorboard --logdir log --bind_all
```
Training parameters could be found at ```config.py```.

