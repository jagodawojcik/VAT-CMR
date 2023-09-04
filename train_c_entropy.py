
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import json
import os
from modality import QueryModality, DominatingModality

from model import TactileNetwork, CrossSensoryNetwork
from load_data import get_loader
from logger import logger


def train_with_cross_entropy(query, dominating_modality, epochs_pre, epochs_c_entropy, batch_size):

    C_ENTROPY_RESULTS_DIRECTORY = os.path.join(f"Triple-CMR-query-{query.value}",f"c-entropy-results-{query.value}-query")

    #Create a directory to save your results
    if os.path.exists(C_ENTROPY_RESULTS_DIRECTORY): 
        raise Exception(f"Directory {C_ENTROPY_RESULTS_DIRECTORY} already exists, please check for existing results.")

    logger.log(f"Directory {C_ENTROPY_RESULTS_DIRECTORY} does not exist, creating...")
    os.makedirs(C_ENTROPY_RESULTS_DIRECTORY)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Training with {device}.")

    
    logger.log("-------Starting pretraining tactile branch.---------")
    # Initialize your Tactile Network
    tactile_network = TactileNetwork().to(device)

    # Initialize your optimizer and loss function for the pretraining
    pretrain_optimizer = torch.optim.Adam(tactile_network.parameters(), lr=0.001)
    pretrain_criterion = nn.CrossEntropyLoss()

    # Get the dataloaders and parameters
    dataloader= get_loader(batch_size)

    # Get the train and test loaders
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    # Initialize list to store losses
    train_losses = []
    test_losses = []

    for epoch in range(epochs_pre):
        tactile_network.train()  # set network to training mode
        total_loss = 0

        for i, (_, tactile_input, _, targets) in enumerate(train_loader):
            tactile_input, targets = tactile_input.to(device), targets.to(device)

            pretrain_optimizer.zero_grad()

            # Get outputs and embeddings
            tactile_output = tactile_network.tactile_branch(tactile_input)
            outputs = tactile_network.fc(tactile_output)

            # Compute the loss
            loss = pretrain_criterion(outputs, targets)
            total_loss += loss.item()

            # Backward and optimize
            loss.backward()
            pretrain_optimizer.step()
            
        # End of epoch
        train_loss = total_loss/len(train_loader)
        train_losses.append(train_loss)
        logger.log(f'Pretraining Epoch {epoch}, Train Loss: {train_loss}')

        # Evaluation loop on test set
        tactile_network.eval()  # set network to evaluation mode
        total_test_loss = 0
        with torch.no_grad():
            for i, (_, tactile_input, _, targets) in enumerate(test_loader):
                tactile_input, targets = tactile_input.to(device), targets.to(device)
                tactile_output = tactile_network.tactile_branch(tactile_input)
                outputs = tactile_network.fc(tactile_output)
                test_loss = pretrain_criterion(outputs, targets)
                total_test_loss += test_loss.item()

        test_loss = total_test_loss/len(test_loader)
        test_losses.append(test_loss)
        logger.log(f'Pretraining Epoch {epoch}, Validation Loss: {test_loss}')

    # Save the model
    torch.save(tactile_network.state_dict(), f"{C_ENTROPY_RESULTS_DIRECTORY}/tactile_model_pretrain.pth")
    
    # Plot train and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.title('Pretraining Tactile Branch: Train and Validation Loss over time', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f"{C_ENTROPY_RESULTS_DIRECTORY}/pretrain_loss_plot.png")
    plt.show()
    

    # Save train and test losses to a JSON file
    loss_dict = {'train_losses': train_losses, 'test_losses': test_losses}
    with open(f"{C_ENTROPY_RESULTS_DIRECTORY}/pretrain_train_test_losses.json", 'w') as f:
        json.dump(loss_dict, f)  # <- Save losses as a JSON file

    logger.log("------Pretraining completed.--------")
    logger.log("------Start cross entropy loss training.--------")

    network = CrossSensoryNetwork(query).to(device)

    # Load the pretrained weights into the tactile branch
    network.tactile_branch.load_state_dict(tactile_network.tactile_branch.state_dict())

    # Initialize optimizer and loss function for the main training
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize lists to store losses
    train_losses = []
    test_losses = []
    dominating_modality_test_losses = []

    # Training loop
    for epoch in range(epochs_c_entropy):
        network.train()  # set network to training mode
        total_train_loss = 0

        # Initialize embeddings storage for each epoch
        audio_embeddings_train = defaultdict(list)
        tactile_embeddings_train = defaultdict(list)
        visual_embeddings_train = defaultdict(list)
        fused_embeddings_train = defaultdict(list)
        
        # Training phase
        for i, (audio_input, tactile_input, visual_input, targets) in enumerate(train_loader):
            audio_input, tactile_input, visual_input, targets = audio_input.to(device), tactile_input.to(device), visual_input.to(device), targets.to(device)

            optimizer.zero_grad()

            # Get outputs and embeddings
            audio_output, tactile_output, visual_output, attention_out, joint_embeddings = network(audio_input, tactile_input, visual_input)

            # Select classification modality
            if dominating_modality == DominatingModality.AUDIO:
                classification_modality = audio_output
            elif dominating_modality == DominatingModality.VISUAL: 
                classification_modality = visual_output
            elif dominating_modality == DominatingModality.TACTILE:
                classification_modality = tactile_output
            else:
                classification_modality = joint_embeddings

            # Compute modality
            loss = criterion(classification_modality, targets)
            total_train_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Save embeddings for each batch
            for j in range(audio_output.shape[0]):
                label = targets[j].item()
                audio_embeddings_train[label].append(audio_output[j].detach().cpu().numpy())
                tactile_embeddings_train[label].append(tactile_output[j].detach().cpu().numpy())
                visual_embeddings_train[label].append(visual_output[j].detach().cpu().numpy())
                fused_embeddings_train[label].append(attention_out[j].detach().cpu().numpy())
        
        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)  # Append training loss for current epoch

        # Evaluation phase on test set
        network.eval()  # set network to evaluation mode
        audio_embeddings_test = defaultdict(list)
        tactile_embeddings_test = defaultdict(list)
        visual_embeddings_test = defaultdict(list)
        fused_embeddings_test = defaultdict(list)

        total_test_loss = 0
        total_test_loss_dominating_modality = 0

        with torch.no_grad():
            for i, (audio_input, tactile_input, visual_input, targets) in enumerate(test_loader):
                audio_input, tactile_input, visual_input, targets = audio_input.to(device), tactile_input.to(device), visual_input.to(device), targets.to(device)

                # Get outputs and embeddings
                audio_output, tactile_output, visual_output, attention_out, joint_embeddings = network(audio_input, tactile_input, visual_input)

                # Compute the loss
                loss = criterion(joint_embeddings, targets)
                total_test_loss += loss.item()
                
                # Compute the loss but for dominating modality
                dominating_mod_loss = criterion(classification_modality, targets)  
                total_test_loss_dominating_modality += dominating_mod_loss.item() 

                # Save test embeddings for each batch
                for j in range(audio_output.shape[0]):
                    label = targets[j].item()
                    audio_embeddings_test[label].append(audio_output[j].detach().cpu().numpy())
                    tactile_embeddings_test[label].append(tactile_output[j].detach().cpu().numpy())
                    visual_embeddings_test[label].append(visual_output[j].detach().cpu().numpy())
                    fused_embeddings_test[label].append(attention_out[j].detach().cpu().numpy())

        test_loss = total_test_loss / len(test_loader)
        test_losses.append(test_loss)  # Append test loss for current epoch
        dominating_modality_test_loss = total_test_loss_dominating_modality / len(test_loader) 
        dominating_modality_test_losses.append(dominating_modality_test_loss)  # dominating modality-specific loss

        logger.log(f'Epoch {epoch}, Train Loss: {epoch_train_loss}, Validation Loss (joint embedding): {test_loss}, {dominating_modality.value} Validation Loss (dominating modality): {dominating_modality_test_loss}')


    # Save the embeddings after all epochs
    logger.log("Training completed. Saving embeddings and model...")
    np.save(f"{C_ENTROPY_RESULTS_DIRECTORY}/audio_embeddings_train.npy", dict(audio_embeddings_train))
    np.save(f"{C_ENTROPY_RESULTS_DIRECTORY}/tactile_embeddings_train.npy", dict(tactile_embeddings_train))
    np.save(f"{C_ENTROPY_RESULTS_DIRECTORY}/visual_embeddings_train.npy", dict(visual_embeddings_train))
    np.save(f"{C_ENTROPY_RESULTS_DIRECTORY}/fused_embeddings_train.npy", dict(fused_embeddings_train))
    np.save(f"{C_ENTROPY_RESULTS_DIRECTORY}/audio_embeddings_test.npy", dict(audio_embeddings_test))
    np.save(f"{C_ENTROPY_RESULTS_DIRECTORY}/tactile_embeddings_test.npy", dict(tactile_embeddings_test))
    np.save(f"{C_ENTROPY_RESULTS_DIRECTORY}/visual_embeddings_test.npy", dict(visual_embeddings_test))
    np.save(f"{C_ENTROPY_RESULTS_DIRECTORY}/fused_embeddings_test.npy", dict(fused_embeddings_test))
    
    # Save the trained model
    torch.save(network.state_dict(), f"{C_ENTROPY_RESULTS_DIRECTORY}/c-entropy-model.pth")
    
    # Save train and test losses to a JSON file
    loss_dict = {'train_losses': train_losses, 'test_losses': test_losses, f'test_{dominating_modality}_losses': dominating_modality_test_losses}
    with open(f"{C_ENTROPY_RESULTS_DIRECTORY}/c_entropy_train_test_losses.json", 'w') as f:
        json.dump(loss_dict, f)  # <- Save losses as a JSON file

    # After training, plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation Loss')

    plt.title('Cross-Entropy Training: Train and Validation Loss over time', fontsize=18)

    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=16)

    plt.show()

    # Save the figure
    plt.savefig(f"{C_ENTROPY_RESULTS_DIRECTORY}/c_entropy_test_loss_plot.png")

    # Display the plot
    plt.show()

