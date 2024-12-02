import socket
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create and bind the server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8080))  # Listen on all interfaces on port 8080
server_socket.listen(1)

print("Waiting for connection...")
conn, addr = server_socket.accept()
print(f"Connected by {addr}")

# Initialize empty lists to store data
epochs = []
accuracies = []
losses = []
val_accuracies = []
val_losses = []

# Initialize global variable
total_epochs = None  # To be set based on incoming data
initial_loss = None  # To store the initial loss for setting y-limits dynamically
initial_val_loss = None # To store the initial validation loss for setting y-limits dynamically

# Set up the plot with 4 subplots (2 rows x 2 columns)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))  # 2x2 grid of subplots
axes = axes.flatten()  # Flatten the axes array to make it easier to index

# Create empty plot objects for each subplot
lines = []
for ax in axes:
    line, = ax.plot([], [], label='Data')  # Initialize each plot with an empty line
    ax.set_xlim(0, 1000)  # Adjust the limit based on your expected number of epochs
    ax.set_ylim(0, 1)  # Adjust the Y-axis based on expected range of data
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.grid(True)
    lines.append(line)

# Add titles and labels to each subplot
axes[0].set_title('Accuracy vs Epoch')
axes[1].set_title('Loss vs Epoch')
axes[2].set_title('Validation Accuracy vs Epoch')
axes[3].set_title('Validation Loss vs Epoch')

axes[0].set_ylim(0, 1)
axes[2].set_ylim(0, 1)
        
# Initialize buffer to accumulate incoming data
buffer = ""

def update(frame):
    global buffer, total_epochs, initial_loss, initial_val_loss
    try:
        # Receive data in chunks
        data = conn.recv(1024).decode()
        if not data:
            return lines  # If no data, just return the current lines
        
        # Append the received chunk to the buffer
        buffer += data

        # Try to parse the JSON from the buffer
        try:
            nn_data = json.loads(buffer)
            buffer = ""  # Reset buffer once the full message is processed
        except json.JSONDecodeError:
            # If not a complete JSON message, keep accumulating data
            return lines

        # Extract data from the parsed JSON
        epoch = nn_data.get("current_epoch", None)
        accuracy = nn_data.get("accuracy", None)
        loss = nn_data.get("loss", None)
        val_accuracy = nn_data.get("val_accuracy", None)
        val_loss = nn_data.get("val_loss", None)

        # Append data to respective lists
        if epoch is not None:
            epochs.append(epoch)
        if accuracy is not None:
            accuracies.append(accuracy)
        if loss is not None:
            losses.append(loss)
        if val_accuracy is not None:
            val_accuracies.append(val_accuracy)
        if val_loss is not None:
            val_losses.append(val_loss)
        
        # Set the x-axis range once based on the number of epochs
        if total_epochs is None and nn_data.get("num_epochs", None) is not None:
            total_epochs = nn_data.get("num_epochs")
            for ax in axes:
                ax.set_xlim(0, total_epochs) 

        if initial_loss is None and nn_data.get("loss", None) is not None:
            initial_loss = loss
            axes[1].set_ylim(0, initial_loss + 1.5 * initial_loss)

        if initial_val_loss is None and nn_data.get("val_loss", None) is not None:
            initial_val_loss = val_loss
            axes[3].set_ylim(0, initial_val_loss + 1.5 * initial_val_loss)

        # Update the lines in each subplot with new data and set distinct colors
        lines[0].set_data(epochs, accuracies)  # Accuracy vs Epoch (Blue)
        lines[0].set_color('blue')
        
        lines[1].set_data(epochs, losses)      # Loss vs Epoch (Red)
        lines[1].set_color('red')
        
        lines[2].set_data(epochs, val_accuracies)  # Validation Accuracy vs Epoch (Blue)
        lines[2].set_color('blue')
        
        lines[3].set_data(epochs, val_losses)  # Validation Loss vs Epoch (Red)
        lines[3].set_color('red')

    except Exception as e:
        print("Error while updating data:", e)

    return lines

# Set up the FuncAnimation to update all subplots
ani = FuncAnimation(fig, update, interval=100, blit=True)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Close the connection and socket when finished
conn.close()
server_socket.close()
