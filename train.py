from model import *
from dataset import *
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import os
import asyncio

def to_jax(tensor):
    return jax.device_put(jnp.array(tensor.detach().cpu().numpy()))

learning_rate = 0.005
momentum = 0.9
batch_size = 64
train_steps = 1200
eval_every = 200
batch_size = 32

model = CNN(rngs=nnx.Rngs(0))

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss')
)

metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}

def loss_fn(model: CNN, X, Y):
    logits = model(X)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=Y).mean()
    return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, X, Y):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, X, Y)
    metrics.update(loss=loss, logits=logits, labels=Y) # In-place updates
    optimizer.update(grads) # In-place updates

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, X, Y):
    loss, logits = loss_fn(model, X, Y)
    metrics.update(loss=loss, logits=logits, labels=Y) # In-place updates

train_dataset = MnistDataset('MNIST_CSV/mnist_train.csv')
test_dataset = MnistDataset('MNIST_CSV/mnist_test.csv')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for step, (X, Y) in enumerate(train_dataloader):
    # Run the optimization for one step and make a stateful update to the folling:
    # - The train state's model parameters
    # - The optimizer state
    # - The training loss and accuracy batch metrics
    train_step(model, optimizer, metrics, to_jax(X.unsqueeze(3)), to_jax(Y))

    if step > 0 and (step % eval_every == 0 or step == train_steps-1): # One training epoch has passed.
        # Log the training metrics.
        for metric, value in metrics.compute().items(): # Compute the metrics.
            metrics_history[f'train_{metric}'].append(value) # Record the metrics.
        metrics.reset() # Reset the metrics for the test set.
    
        # Compute the metrics on the test set after each training epoch.
        for X_test, Y_test in test_dataloader:
            eval_step(model, metrics, to_jax(X.unsqueeze(3)), to_jax(Y))

        # Log the test metrics.
        for metric, value in metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)
            metrics.reset() # Reset the metrics for the next training epoch.

        print({key:metrics_history[key][-1] for key in metrics_history})
        '''
        # Plot loss and accuracy in subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title("Loss")
        ax2.set_title("Accuracy")
        for dataset in ("train", "test"):
            ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
            ax2.plot(metrics_history[f"{dataset}_accuracy"], label=f'{dataset}_accuracy')
        ax1.legend()
        ax2.legend()
        plt.show()
        '''

def save_checkpoint(ckpt_dir, state, step=None):
    """Simple checkpoint function using the correct Orbax API."""
    try:
        # Create checkpoint directory
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Create a handler for nnx state
        handler = ocp.PyTreeCheckpointHandler()
        
        # Create a checkpointer with the handler
        checkpointer = ocp.Checkpointer(handler)
        
        # Configure options
        options = ocp.CheckpointManagerOptions(
            max_to_keep=3,
            create=True,
        )
        
        # Create manager
        manager = ocp.CheckpointManager(
            directory=ckpt_dir,
            checkpointers={"model_state": checkpointer},
            options=options,
        )
        
        # Get step number
        if step is None:
            step = state.step if hasattr(state, 'step') else 0
        
        # Save the checkpoint
        manager.save(step, {"model_state": state})
        
        print(f"Checkpoint saved at step {step}")
        return True
    except RuntimeError as e:
        if "cannot schedule new futures after" in str(e):
            print("Warning: Checkpoint interrupted by shutdown")
            return False
        raise

_, state = nnx.split(model)
ckpt_dir = '/home/remy/mnist_flax/checkpoints'
save_checkpoint(ckpt_dir, state, step)
