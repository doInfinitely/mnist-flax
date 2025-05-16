import orbax.checkpoint as ocp
from model import *
from dataset import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

def to_jax(tensor):
    return jax.device_put(jnp.array(tensor.detach().cpu().numpy()))

# Use the fixed loading function
def load_checkpoint(ckpt_dir):
    """
    Load the latest checkpoint using the new Orbax API.
    
    Args:
        ckpt_dir: Directory where checkpoints are saved
    
    Returns:
        A tuple of (loaded_state, step_number) if successful
        (None, None) if no checkpoint was found or if loading failed
    """
    try:
        # Create the same handler used for saving
        handler = ocp.PyTreeCheckpointHandler()
        
        # Create a checkpointer with the handler
        checkpointer = ocp.Checkpointer(handler)
        
        # Create manager with the same structure as used for saving
        manager = ocp.CheckpointManager(
            directory=ckpt_dir,
            checkpointers={"model_state": checkpointer},
        )
        
        # Get the latest step (or None if no checkpoints exist)
        latest_step = manager.latest_step()
        
        if latest_step is None:
            print(f"No checkpoints found in {ckpt_dir}")
            return None, None
        
        # Restore the checkpoint without providing a target
        # This will just load whatever structure was saved
        restored = manager.restore(latest_step, {"model_state": None})
        
        # Get the loaded state
        loaded_state = restored["model_state"]
        
        print(f"Checkpoint successfully loaded from step {latest_step}")
        
        return loaded_state, latest_step
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Create a new model first
ckpt_dir = '/home/remy/mnist_flax/checkpoints'
model = CNN(rngs=nnx.Rngs(0))

# Split to get the graph definition
graphdef, _ = nnx.split(model)

# Load the state without trying to apply it to a target
state, step = load_checkpoint(ckpt_dir)
if state is not None:
    print('NNX State restored. Creating model...')
    
    # Merge the graph with the loaded state
    try:
        model = nnx.merge(graphdef, state)
        print('Model successfully created from checkpoint!')
        
        # Switch to evaluation mode
        model.eval()
        
        # Define prediction function
        @nnx.jit
        def pred_step(model, X):
            logits = model(X)
            return logits.argmax(axis=1)
        
        # Load test data
        test_dataset = MnistDataset('MNIST_CSV/mnist_test.csv')
        batch_size = 32
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # Make predictions on a batch
        for X, Y in test_dataloader:
            X_jax = to_jax(X.unsqueeze(3))
            pred = pred_step(model, X_jax)
            
            # Print accuracy
            Y_jax = to_jax(Y)
            accuracy = (pred == Y_jax).mean()
            print(f"Test accuracy: {accuracy:.4f}")
            
            # Visualize results
            fig, axs = plt.subplots(5, 5, figsize=(12, 12))
            for i, ax in enumerate(axs.flatten()):
                if i < len(X):
                    ax.imshow(X[i], cmap='gray')
                    ax.set_title(f'Pred: {pred[i]}, True: {Y[i]}')
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig("inference.png", dpi=300)
            plt.show()
            break
    except Exception as e:
        print(f"Error creating model from state: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Failed to load checkpoint state")
