this is an open source implementation of Chameleon: Mixed-Modal Early-Fusion Foundation Models(arXiv:2405.09818v1) from meta. designed to run on a subset of mscoco on cpu for a small scale model, with both multimodal inputs and outputs.

to run a small scale training on cpu;
firstly, this code uses custom version of pytorch that is cerebras.pytorch for training script, but model and other code written in standart pytorch.
for training i recommend a virtual environment in python3.8 version.
while running this, you may need to install some libraries such as "pip install cerebras_pytorch".

```python
python -m venv chameleon
source chameleon/bin/activate
git clone https://github.com/notlober/chameleon.git
cd chameleon
python prepare_mini_coco.py
```

this will get you the gpt4 tokenizer from tiktoken library from openai (open source tokenizer),
768 static image tokens that represents each pixel in R, G, B channels, each as a disrete text token,
also this will get you the train.bin file, which includes 50 image text pair as tokenized for autoregressive training.
this model gets input as images and text, returns image and text as tokens back.

then;

```python
python train.py config.yaml
```

this will start training and output loss values in terminal for each step. 
you can further experiment by modifying config.yaml file or by forking repo and modifying code.
for now it only trains for 5 steps, the output is like, becomes stable between 6-7;

```python
INFO:train.py:
Step=1, Loss=11.67453
INFO:train.py:
Step=2, Loss=11.74671
INFO:train.py:
Step=3, Loss=10.34042
INFO:train.py:
Step=4, Loss=8.97504
INFO:train.py:
Step=5, Loss=7.32438
```
