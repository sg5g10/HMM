import torch
from torch.utils import data  
from torch import nn, optim

from torch.nn.utils.clip_grad import clip_grad_norm_  
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
from copy import deepcopy
from neural_net.maf import MaskedAutoregressiveTransform, RandomPermutation, CompositeTransform
from neural_net.flow_utils import standardizing_transform, standardizing_net
from neural_net.flow import Flow, StandardNormal
import time

def build_maf(targ, cond, targ_dim, cond_dim, 
            hidden_features = 50, num_transforms = 5,
            num_blocks = 2):
   
    transform_list = []
    for _ in range(num_transforms):
        block = [
            MaskedAutoregressiveTransform(
                features=targ_dim,
                hidden_features=hidden_features,
                context_features=cond_dim,
                num_blocks=num_blocks,
                use_residual_blocks=False,
                activation=F.tanh),
            RandomPermutation(features=targ_dim),
        ]
        transform_list += block

 
    transform_list = [
            standardizing_transform(targ)
        ] + transform_list


    embedding_net = nn.Sequential(
            standardizing_net(cond), nn.Identity()
        )

    # Combine transforms.
    transform = CompositeTransform(transform_list)

    distribution = StandardNormal((targ_dim,))
    neural_net = Flow(transform, distribution, embedding_net)

    return neural_net

def get_dataloaders(dataset, training_batch_size= 50, validation_fraction = 0.1):
       
        num_examples = len(dataset)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        permuted_indices = torch.randperm(num_examples)
        train_indices, val_indices = (
                permuted_indices[:num_training_examples],
                permuted_indices[num_training_examples:],
            )

        # Create training and validation loaders using a subset sampler.
        # Intentionally use dicts to define the default dataloader args
        # Then, use dataloader_kwargs to override (or add to) any of these defaults
        # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "sampler": SubsetRandomSampler(train_indices.tolist()),
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            "sampler": SubsetRandomSampler(val_indices.tolist()),
        }

        train_loader = data.DataLoader(dataset, **train_loader_kwargs)
        val_loader = data.DataLoader(dataset, **val_loader_kwargs)

        return train_loader, val_loader, train_indices, val_indices 


def train(
    targ,
    cond,
    training_batch_size = 256,
    learning_rate = 5e-4,
    validation_fraction = 0.1,
    stop_after_epochs = 20,
    max_num_epochs: int = 2**31 - 1,
    clip_max_norm = 5.0,
    device="cpu"):        

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(targ, cond)

        train_loader, val_loader, train_indices, _  = get_dataloaders(dataset,
            training_batch_size,
            validation_fraction)

        neural_net = build_maf(targ[train_indices], cond[train_indices], targ.shape[1], cond.shape[1] )
        neural_net.to(device)

        optimizer = optim.Adam(
            list(neural_net.parameters()), lr=learning_rate
        )
        epoch, val_log_prob, best_val_log_prob = 0, float("-Inf"), 0
        converged = False
        epochs_since_last_improvement = 0
        while epoch <= max_num_epochs and not converged:
            # Train for a single epoch.
            neural_net.train()
            train_log_probs_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                optimizer.zero_grad()
                # Get batches on current device.
                targ_batch, cond_batch = (
                    batch[0].to(device),
                    batch[1].to(device)
                )

                train_losses = -neural_net.log_prob(
                    targ_batch, cond_batch)
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        neural_net.parameters(), max_norm=clip_max_norm
                    )
                optimizer.step()

              

            # Calculate validation performance.
            neural_net.eval()
            val_log_prob_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(device),
                        batch[1].to(device))
                    # Take negative loss here to get validation log_prob.
                    val_losses = -neural_net.log_prob(
                        theta_batch,
                        x_batch)
                    val_log_prob_sum -= val_losses.sum().item()

            # Take mean over all validation samples.
            val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
        

        # (Re)-start the epoch count with the first epoch or any improvement.
            
            if epoch == 0 or val_log_prob > best_val_log_prob:
                best_val_log_prob = val_log_prob
                epochs_since_last_improvement = 0
                best_model_state_dict = deepcopy(neural_net.state_dict())
            else:
                epochs_since_last_improvement += 1

            # If no validation improvement over many epochs, stop training.
            if epochs_since_last_improvement > stop_after_epochs - 1:
                neural_net.load_state_dict(best_model_state_dict)
                converged = True

            epoch += 1   
            
            print("\r", f"Training neural network. Epochs trained: {epoch}", end="")


         # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        print("\r", f"Neural network converged after {epoch} epochs", end="")
        neural_net.zero_grad(set_to_none=True)

        return deepcopy(neural_net)