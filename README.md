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

## Evaluation of the AI

To assess the performance of the AI, run this command:

```bash
python main.py benchmark
```

It will provide you with the accuracy for each class and the overall accuracy on the test dataset.

## Perform Adversarial Attack in Real Time

```bash
python main.py draw
```

Note that if your PC is too slow, you can run this command instead:

```bash
python main.py draw --optim
```

This will add a button `predict`; the digit will only be predicted when this button is pressed.

Once the AI recognizes the digit you have drawn, press any digit on your keyboard to perform the adversarial attack. The right noise, which must be added to the image, will then be computed. If you just want the prediction to change to any other number, press the `a` key (for `any`).