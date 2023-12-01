# MNIST Adversarial Attack

This is an example of an adversarial attack performed on MNIST.

## Install Requirements

You might have to install `tkinter` yourself. To get the other requirements, run:

```bash
pip install -r requirements.txt
```

## Downloading the Dataset

```bash
python main.py download
```

## Visualizing the Dataset

```bash
python main.py plot
```

## Training the AI

```bash
python main.py train --batch 64 --epoch 10 --lr 0.001
```

## Perform Adversarial Attack in Real Time

```bash
python main.py draw
```

Note that if your PC is too slow, you can run this command instead:

```bash
python main.py draw --optim
```

This will add a button `predict`, the digit will only be predicted when this button is pressed.