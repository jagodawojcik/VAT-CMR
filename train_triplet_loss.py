from evaluation import evaluate
from logger import logger
from model import EmbeddingNet, TripletLoss, TripletDataset

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pathlib

def train_with_triplet_loss(query, dominating, epochs, batch_size, margin):
    
    # Create a directory to save your results
    TRIPLET_RESULTS_DIRECTORY = os.path.join(f"Triple-CMR-query-{query.value}-dom-{dominating.value}",f"triplet-results-{query.value}-query")

    #Create a directory to save your results
    if os.path.exists(TRIPLET_RESULTS_DIRECTORY): 
        raise Exception(f"Directory {TRIPLET_RESULTS_DIRECTORY} already exists, please search for previous results before running.")

    logger.log(f"Directory {TRIPLET_RESULTS_DIRECTORY} does not exist, creating...")
    os.makedirs(TRIPLET_RESULTS_DIRECTORY)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Training with {device}.")

    CURRENT_DIRECTORY = pathlib.Path(__file__).parent.resolve()
    EMBEDDINGS_DIRECTORY = os.path.join(CURRENT_DIRECTORY, ".." , f"Triple-CMR-query-{query.value}-dom-{dominating.value}", f'c-entropy-results-{query.value}-query')

    # Load your embeddings
    query_embeddings = np.load(os.path.join(EMBEDDINGS_DIRECTORY, f"{query.value}_embeddings_train.npy"), allow_pickle=True).item()
    fused_embeddings = np.load(os.path.join(EMBEDDINGS_DIRECTORY, "fused_embeddings_train.npy"), allow_pickle=True).item()
    query_embeddings_test = np.load(os.path.join(EMBEDDINGS_DIRECTORY,f"{query.value}_embeddings_test.npy"), allow_pickle=True).item()
    fused_embeddings_test = np.load(os.path.join(EMBEDDINGS_DIRECTORY, "fused_embeddings_test.npy"), allow_pickle=True).item()

    # Instantiate your dataset and dataloader
    triplet_dataset = TripletDataset(query_embeddings, fused_embeddings)
    triplet_dataloader = DataLoader(triplet_dataset, batch_size, shuffle=True)

    # Initialize loss function
    triplet_loss = TripletLoss(margin)
    
    # Initialize model
    model = EmbeddingNet(embedding_dim=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Create a directory to save your results
    results_map = {
        'fused2query': [],
        'query2fused': []
    }
    triplet_loss_save = {
        'triplet_loss': []
    }
    best_map_pairs = {
        'MAP_pairs': []
    }

    # Initialize max MAP values to get best MAP results during training
    max_query2fused = 0.0
    max_fused2query = 0.0
    max_MAP_total = 0.0
    result_epoch = 0

    # Start training loop
    for epoch in range(epochs):
        total_loss = 0

        for i, (anchor, positive, negative, label) in enumerate(triplet_dataloader):
            # Move to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Pass data through the model
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            # Compute the loss
            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Add loss
            total_loss += loss.item()

        avg_loss = total_loss / len(triplet_dataloader)
        triplet_loss_save['triplet_loss'].append(avg_loss)
        # logger.log('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, avg_loss))

        if epoch % 100 == 0:
            new_query_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in query_embeddings.items()}
            new_fused_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in fused_embeddings.items()}
            
            with torch.no_grad():
                new_query_embeddings_test = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in query_embeddings_test.items()}
                new_fused_embeddings_test = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in fused_embeddings_test.items()}

            # Evaluate embeddings with validation dataset
            MAP_fused2query, MAP_query2fused = evaluate(new_query_embeddings_test, new_fused_embeddings_test, new_query_embeddings, new_fused_embeddings)
            
            if (MAP_fused2query + MAP_query2fused) > max_MAP_total:
                max_fused2query = MAP_fused2query
                max_query2fused = MAP_query2fused
                best_map_pairs['MAP_pairs'].append((epoch, MAP_fused2query, MAP_query2fused))
                np.save('{}/triplet_trained_retrieval_query_embeddings.npy'.format(TRIPLET_RESULTS_DIRECTORY), new_query_embeddings)
                np.save('{}/triplet_trained_retrieval_fused_embeddings'.format(TRIPLET_RESULTS_DIRECTORY), new_fused_embeddings)
                torch.save(model.state_dict(), f"{TRIPLET_RESULTS_DIRECTORY}/triplet_model_best.pth")
                result_epoch = epoch
            # if MAP_fused2query > max_fused2query:
            #     max_fused2query = MAP_fused2query
            #     best_map_pairs['MAP_pairs'].append((epoch, MAP_fused2query, MAP_query2fused))
            #     np.save('{}/trained_query_embeddings_{}.npy'.format(TRIPLET_RESULTS_DIRECTORY, epoch), new_query_embeddings)
            #     np.save('{}/trained_fused_embeddings_{}.npy'.format(TRIPLET_RESULTS_DIRECTORY, epoch), new_fused_embeddings)
            #     torch.save(model.state_dict(), f"{TRIPLET_RESULTS_DIRECTORY}/model_best_fused2query.pth")
                
            # if MAP_query2fused > max_query2fused:
            #     max_query2fused = MAP_query2fused
            #     best_map_pairs['MAP_pairs'].append((epoch, MAP_fused2query, MAP_query2fused))
            #     np.save('{}/trained_query_embeddings_{}.npy'.format(TRIPLET_RESULTS_DIRECTORY, epoch), new_query_embeddings)
            #     np.save('{}/trained_fused_embeddings_{}.npy'.format(TRIPLET_RESULTS_DIRECTORY, epoch), new_fused_embeddings)
            #     torch.save(model.state_dict(), f"{TRIPLET_RESULTS_DIRECTORY}/model_best_query2fused.pth")

            # Add the results to the map
            results_map['fused2query'].append(MAP_fused2query)
            results_map['query2fused'].append(MAP_query2fused)

    # Save the map results as a JSON file
    with open('{}/map_results_{}.json'.format(TRIPLET_RESULTS_DIRECTORY, epoch), 'w') as f:
        json.dump(results_map, f)
    with open('{}/triplet_loss.json'.format(TRIPLET_RESULTS_DIRECTORY), 'w') as f:
        json.dump(triplet_loss_save, f)
    with open('{}/best_map_pairs.json'.format(TRIPLET_RESULTS_DIRECTORY), 'w') as f:
        json.dump(best_map_pairs, f)

    # Plot the results
    plt.figure(figsize=(12,6))
    plt.plot(range(len(results_map['fused2query'])), results_map['fused2query'], label='Fused to Query')
    plt.plot(range(len(results_map['query2fused'])), results_map['query2fused'], label='Query to Fused')
    plt.xlabel('Triplet Loss Training Epoch')
    plt.ylabel('MAP')
    plt.legend()
    plt.title('MAP Results - Triplet Loss Training')
    plt.savefig('{}/map_plot_{}.png'.format(TRIPLET_RESULTS_DIRECTORY, epoch))
    plt.close()

    #Print best results and save them to an information file
    logger.log('MAP Fused to Query: {}'.format(max_fused2query))
    logger.log('MAP Query to Fused: {}'.format(max_query2fused))

    with open(f"{TRIPLET_RESULTS_DIRECTORY}/MAP_validation_results.txt", "w") as file:
        # Write the user's input to the file
        file.write(f"\n Validation MAP Fused to Query: {max_fused2query}")
        file.write(f"\n Validation MAP Query to Fused: {max_query2fused}")
        file.write(f"\n Saved at Epoch: {result_epoch}")

    # Plot the triplet loss
    plt.figure(figsize=(12,6))
    plt.plot(range(len(triplet_loss_save['triplet_loss'])), triplet_loss_save['triplet_loss'], label='Triplet Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Triplet Loss', fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Triplet Loss Training', fontsize=18)
    plt.savefig(f'{TRIPLET_RESULTS_DIRECTORY}/triplet_loss_plot.png')
    plt.close()


if __name__ == '__main__':
    train_with_triplet_loss()


