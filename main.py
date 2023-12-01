import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm
import torch
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import utils
from topology import CNN
import preprocess

try:
    import tkinter as tk
    from PIL import Image, ImageTk
    from PIL import Image, ImageDraw
except ImportError:
    graphics_installed = False
    print("Warning: tkinter and/or PIL not installed. You will not be able to draw digits.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device : {device}")

def plot(args):
    train_images, train_labels, test_images, test_labels = utils.load()
    
    # Select n random images
    random_indices = np.random.choice(len(train_images), size=args.n, replace=False)
    
    images_to_plot = torch.tensor(train_images[random_indices]).unsqueeze(dim=1)
    if args.normalize:
        images_to_plot = preprocess.normalize(images_to_plot)
    if args.rotate:
        images_to_plot = preprocess.rotate(images_to_plot)
    if args.translate:
        images_to_plot = preprocess.translate(images_to_plot)
    images_to_plot = images_to_plot.squeeze(dim=1).numpy()
    labels_to_plot = train_labels[random_indices]
    
    # Calculate the number of rows and columns for the grid
    num_rows = int(np.ceil(np.sqrt(args.n)))
    num_cols = int(np.ceil(args.n / num_rows))
    
    # Create a grid of images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    fig.suptitle(f"Randomly Selected MNIST Images (n={args.n})", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < args.n:
            ax.imshow(images_to_plot[i], cmap="gray")
            ax.axis("off")
            ax.set_title(labels_to_plot[i])
    
    plt.tight_layout()
    plt.show()
    
def download(args):
    utils.download()

def benchmark(args):
    model = CNN()
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    
    model = model.to(device)
    model.eval()
    
    train_images, train_labels, test_images, test_labels = utils.load()
    
    test_images = preprocess.normalize(torch.tensor(test_images, dtype=torch.float32).unsqueeze(dim=1)).squeeze(dim=1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    test_images = test_images.unsqueeze(dim=1)
    
    for i in range(10):
        #compute the accuracy of the model fo class i
        indices = test_labels == i
        images = test_images[indices]
        labels = test_labels[indices]
        pred = model(images)
        pred = pred.argmax(dim=1)
        accuracy = (pred == labels).sum().item() / len(labels)
        print(f"Test accuracy for class {i}: {accuracy}")
    
    pred = model(test_images)
    pred = pred.argmax(dim=1)
    accuracy = (pred == test_labels).sum().item() / len(test_labels)
    print(f"Test accuracy: {accuracy}")
    
    

def train(args):
    model = CNN()
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    
    model = model.to(device)
    
    train_images, train_labels, test_images, test_labels = utils.load()
    
    # Convert the training data to PyTorch tensors
    train_images = torch.tensor(train_images, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    # Create a dataset from the training data
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    
    # Create a data loader for the training data
    train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    test_images = preprocess.normalize(torch.tensor(test_images, dtype=torch.float32).unsqueeze(dim=1)).squeeze(dim=1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    test_images = test_images.unsqueeze(dim=1)
    
    # Iterate over the batches of training data
    for epoch in range(args.epoch):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch_images, batch_labels in tqdm(train_loader):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device) 
            
            batch_images = preprocess.normalize(batch_images.unsqueeze(dim=1))
            batch_images = preprocess.rotate(batch_images)
            batch_images = preprocess.translate(batch_images)
            pred = model(batch_images)
            
            loss = criterion(pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        pred = model(test_images)
        pred = pred.argmax(dim=1)
        accuracy = (pred == test_labels).sum().item() / len(test_labels)
        print(f"Test accuracy: {accuracy}")
    
    torch.save(model.state_dict(), "model.pth")


def draw_digit(args):

    if not graphics_installed:
        raise ImportError("tkinter and/or PIL not installed. You will not be able to draw digits.")

    model = CNN()
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    
    model = model.to(device)
    model.eval()
    
    # Create a tkinter window
    window = tk.Tk()
    window.title("Draw Digit")
    
    # Create a canvas to draw on
    canvas_width = args.size
    canvas_height = args.size
    cell_size = canvas_width // 28
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="black")
    canvas.pack()
    
    # Function to handle mouse movement
    canvas.array = np.zeros((28, 28))
    
    def on_mouse_move(event):
        col = event.x // cell_size
        row = event.y // cell_size
        
        point = np.zeros((28, 28))
        if 0 <= row < 28 and 0 <= col < 28:
            point[row, col] = 1.0
        else:
            return
        #apply a gaussian blur to the point
        point = gaussian_filter(point, sigma=0.9)
        point = point / point.max()
        
        indices = point < 0.1
        point[indices] = 0
        
        canvas.array = np.clip(canvas.array + point, 0, 1)
        
        update(~indices)
        
        if not args.optim:
            predict_digit()
    
    # Function to predict the digit

    def update(indices):
        
        for row in range(28):
            for col in range(28):
                if not indices[row, col]:
                    continue
                gray_value = int(255 * canvas.array[row, col])  # Adjust this value based on your desired intensity
                
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = (col + 1) * cell_size
                y2 = (row + 1) * cell_size

                # Create a rectangle with the specified grayscale color
                color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

    def predict_digit():
        tensor_image = torch.from_numpy(canvas.array).unsqueeze(dim=0).unsqueeze(dim=0).float()
        prediction = model(tensor_image)
        predicted_digit = prediction.argmax(dim=1).item()
        
        # Display the predicted digit
        result_label.config(text=f"Predicted Digit: {predicted_digit}")
        return predicted_digit
    
    def clear_drawing():
        canvas.delete("all")
        canvas.array = np.zeros((28, 28))
    
    def predict(event):
        digit = event.char
        if not digit.isdigit():
            if event.char.lower() == "a":
                digit = -1
            else:
                return
        else:
            digit = int(digit)
        tensor_image = torch.from_numpy(canvas.array).unsqueeze(dim=0).unsqueeze(dim=0).float()
        
        noise = 0.1*torch.randn_like(tensor_image)
        noise.requires_grad = True
        
        sigmoid = torch.nn.Sigmoid()
        
        old_digit = predict_digit()
        pred = old_digit

        max_iter = 100
        digit_tensor = torch.zeros(10)
        if digit != -1:
            digit_tensor[old_digit] = 0.1
            digit_tensor[digit] = -1
            admissible_digit = [digit]
        else:
            digit_tensor[old_digit] = 1
            admissible_digit = [i for i in range(10) if i != old_digit]
        max_noise = 1
        
        flag = False
        for max_noise in tqdm(np.linspace(0.2, 1, 10)):
            if flag:
                break
            
            optimizer = torch.optim.Adam([noise], lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
            for _ in range(max_iter):
                scale_noise = max_noise*2*(sigmoid(2*noise) - 1/2)
                candidate_image = sigmoid(4*(tensor_image + scale_noise - 1/2))
                prediction = model(candidate_image)
                pred = prediction.argmax(dim=1).item()
                optimizer.zero_grad()
                loss = torch.sum(digit_tensor*prediction)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(noise, 0.1)
                optimizer.step()
                scheduler.step()
                
                flag = pred in admissible_digit
                if flag:
                    break
        
        #set the image on the canvas equal to the gradient
        predicted_digit = prediction.argmax(dim=1).item()
        
        # Display the predicted digit
        result_label.config(text=f"Predicted Digit: {predicted_digit}")

        new_image = candidate_image.squeeze(dim=0).squeeze(dim=0).detach().numpy()

        if args.plot:
            #draw old, new image and noise on the same figure
            fig, axes = plt.subplots(1, 3, figsize=(10, 10))
            fig.suptitle(f"Old, New and Noise Image", fontsize=16)
            axes[0].imshow(canvas.array, cmap="gray")
            axes[0].axis("off")
            axes[0].set_title(old_digit)
            axes[1].imshow(new_image, cmap="gray")
            axes[1].axis("off")
            axes[1].set_title(predicted_digit)
            axes[2].imshow(scale_noise.squeeze(dim=0).squeeze(dim=0).detach().numpy(), cmap="gray")
            axes[2].axis("off")
            axes[2].set_title("Noise")
            plt.tight_layout()
            plt.show()
        
        canvas.array = new_image
        update(canvas.array > -1)
        
        
    # Button to predict the digit
    if args.optim:
        predict_button = tk.Button(window, text="Predict", command=predict_digit)
        predict_button.pack()

    # Button to clear the drawing
    clear_button = tk.Button(window, text="Clear", command=clear_drawing)
    clear_button.pack()
    
    # Label to display the predicted digit
    result_label = tk.Label(window, text="")
    result_label.pack()
    
    # Bind mouse events to canvas
    canvas.bind("<B1-Motion>", on_mouse_move)
    #bind predict to any digit key press
    window.bind("<Key>", predict)
    
    
    # Run the tkinter event loop
    window.mainloop()

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quickdraw")
    
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    
    plot_parser = subparsers.add_parser("plot", help="visualize the data")
    plot_parser.add_argument("-n", type=int, default=12, help="number of images to plot")
    plot_parser.add_argument("--normalize", action="store_true", help="normalize the images")
    plot_parser.add_argument("--rotate", action="store_true", help="rotate the images")
    plot_parser.add_argument("--translate", action="store_true", help="translate the images")
    plot_parser.set_defaults(func=plot)
    
    download_parser = subparsers.add_parser("download", help="download the data")
    download_parser.set_defaults(func=download)
    
    train_parser = subparsers.add_parser("train", help="train the model")
    train_parser.add_argument("--batch", type=int, default=256, help="batch size")
    train_parser.add_argument("--epoch", type=int, default=5, help="number of epochs")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    train_parser.set_defaults(func=train)
    
    draw_parser = subparsers.add_parser("draw", help="draw a digit")
    draw_parser.add_argument("--size", type=int, default=280, help="size of the canvas")
    draw_parser.add_argument("--optim", action="store_true", help="only predict when pressing the predict button")
    draw_parser.add_argument("--plot", action="store_true", help="plot the noise use for adversarial attack")
    draw_parser.set_defaults(func=draw_digit)
    
    benchmark_parser = subparsers.add_parser("benchmark", help="benchmark the model")
    benchmark_parser.set_defaults(func=benchmark)
    
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()