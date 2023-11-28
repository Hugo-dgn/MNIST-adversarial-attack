import numpy as np
from tqdm.auto import tqdm
import torch
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import utils
from topology import CNN

try:
    import tkinter as tk
    from PIL import Image, ImageTk
    from PIL import Image, ImageDraw
except ImportError:
    graphics_installed = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device : {device}")

def plot(args):
    train_images, train_labels, test_images, test_labels = utils.load()
    
    # Select n random images
    random_indices = np.random.choice(len(train_images), size=args.n, replace=False)
    images_to_plot = normalize(torch.tensor(train_images[random_indices]).unsqueeze(dim=1)).squeeze(dim=1).numpy()
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

def train(args):
    model = CNN()
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    
    train_images, train_labels, test_images, test_labels = utils.load()
    
    # Convert the training data to PyTorch tensors
    train_images = torch.tensor(train_images, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    # Create a dataset from the training data
    dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    
    # Create a data loader for the training data
    train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Iterate over the batches of training data
    for batch_images, batch_labels in tqdm(train_loader):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        batch_images = normalize(batch_images.unsqueeze(dim=1))
        pred = model(batch_images)
        
        loss = criterion(pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #test 
    test_images = normalize(torch.tensor(test_images, dtype=torch.float32).unsqueeze(dim=1)).squeeze(dim=1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    test_images = test_images.unsqueeze(dim=1)
    pred = model(test_images)
    pred = pred.argmax(dim=1)
    accuracy = (pred == test_labels).sum().item() / len(test_labels)
    print(f"Test accuracy: {accuracy:.2f}")
    
    torch.save(model.state_dict(), "model.pth")

def normalize(images):
    
    min_values = torch.amin(images, dim=(2, 3), keepdim=True)
    max_values = torch.amax(images, dim=(2, 3), keepdim=True)

# Subtract the minimum values and normalize
    normalized_image = (images - min_values) / (max_values - min_values + 1e-6)
    return normalized_image


def draw_digit(args):

    model = CNN()
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    
    model = model.to(device)
    model.eval()
    
    # Create a tkinter window
    window = tk.Tk()
    window.title("Draw Digit")
    
    # Create a canvas to draw on
    canvas_width = 200
    canvas_height = 200
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="black")
    canvas.pack()
    
    # Create an image and draw object
    image = Image.new("L", (canvas_width, canvas_height), color=0)
    draw = ImageDraw.Draw(image)
    
    # Variables to track mouse movement
    prev_x = None
    prev_y = None
    
    # Function to handle mouse movement
    
    def on_mouse_move(event):
        nonlocal prev_x, prev_y
        x = event.x
        y = event.y
        if prev_x is not None and prev_y is not None:
            canvas.create_line(prev_x, prev_y, x, y, width=10, fill="white")
            draw.line([(prev_x, prev_y), (x, y)], fill=255, width=10)
        prev_x = x
        prev_y = y
    
    # Function to handle mouse release
    def on_mouse_release(event):
        nonlocal prev_x, prev_y
        prev_x = None
        prev_y = None
    
    # Function to predict the digit
    
    def preprocess(image):
        resized_image = image.resize((28, 28))
        
        # Convert the image to grayscale and normalize the pixel values
        grayscale_image = resized_image.convert("L")
        grayscale_image = np.array(grayscale_image)
        
        # Convert the image to a PyTorch tensor
        tensor_image = torch.tensor(grayscale_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_image = normalize(tensor_image)
        
        return tensor_image
    def predict_digit():
        #get the image on the canvas
        tensor_image = preprocess(image)
        # Feed the image to the CNN model and predict the digit
        prediction = model(tensor_image)
        predicted_digit = prediction.argmax(dim=1).item()
        
        # Display the predicted digit
        result_label.config(text=f"Predicted Digit: {predicted_digit}")
    
    def clear_drawing():
        canvas.delete("all")
        image.paste(0, box=(0, 0, 200, 200))
        draw.rectangle((0, 0, 200, 200), fill=0)
    
    def predict(event):
        digit = event.char
        if not digit.isdigit():
            return
        else:
            digit = int(digit)
        tensor_image = preprocess(image)
        tensor_image.requires_grad_(True)
        
        optimizer = torch.optim.Adam([tensor_image], lr=1e-2)
        
        prediction = model(tensor_image)
        while prediction.argmax(dim=1).item() != digit:
            optimizer.zero_grad()
            loss = -prediction[0][digit]
            loss.backward()
            optimizer.step()
            prediction = model(tensor_image)
        
        #set the image on the canvas equal to the gradient
        predicted_digit = prediction.argmax(dim=1).item()
        
        # Display the predicted digit
        result_label.config(text=f"Predicted Digit: {predicted_digit}")
        
        #transform the image back to a PIL image
        new_image = np.clip(255*tensor_image.squeeze(dim=0).squeeze(dim=0).detach().numpy(), 0, 255)
        new_image = Image.fromarray(new_image)
        new_image = new_image.resize((200, 200))
        #paste the new image on the canvas
        image.paste(new_image, box=(0, 0, 200, 200))
        #draw.rectangle((0, 0, 200, 200), fill=0)
        
        canvas.delete("all")
        canvas.image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=canvas.image)
        
        
        
        

    # Button to clear the drawing
    clear_button = tk.Button(window, text="Clear", command=clear_drawing)
    clear_button.pack()
    
    # Button to predict the digit
    predict_button = tk.Button(window, text="Predict", command=predict_digit)
    predict_button.pack()
    
    # Label to display the predicted digit
    result_label = tk.Label(window, text="")
    result_label.pack()
    
    # Bind mouse events to canvas
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_release)
    #bind predict to any digit key press
    window.bind("<Key>", predict)
    
    
    # Run the tkinter event loop
    window.mainloop()

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quickdraw")
    
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    
    plot_parser = subparsers.add_parser("plot", help="visualize the data")
    plot_parser.add_argument("-n", type=int, default=10, help="number of images to plot")
    plot_parser.set_defaults(func=plot)
    
    download_parser = subparsers.add_parser("download", help="download the data")
    download_parser.set_defaults(func=download)
    
    train_parser = subparsers.add_parser("train", help="train the model")
    train_parser.add_argument("--batch", type=int, default=64, help="batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    train_parser.set_defaults(func=train)
    
    draw_parser = subparsers.add_parser("draw", help="draw a digit")
    draw_parser.set_defaults(func=draw_digit)
    
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()