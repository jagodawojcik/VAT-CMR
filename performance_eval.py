import torch
import numpy as np
from collections import defaultdict
import os

from logger import logger
from load_data import get_test_loader  
from model import CrossSensoryNetwork, EmbeddingNet
from evaluation import evaluate

def test_set_performance_evaluate(query, dominating_modality):

    logger.log("Start final performance evaluation on a test set.")
    
    # Initialize results directory
    FINAL_EVALUATION_RESULTS_DIRECTORY = os.path.join(f"Triple-CMR-query-{query.value}-dom-{dominating_modality.value}",f"performance-evaluation-results")

    #Create a directory to save your results
    if os.path.exists(FINAL_EVALUATION_RESULTS_DIRECTORY): 
        raise Exception(f"Directory {FINAL_EVALUATION_RESULTS_DIRECTORY} already exists, please check for existing results.")
    
    logger.log(f"Directory {FINAL_EVALUATION_RESULTS_DIRECTORY} does not exist, creating...")
    os.makedirs(FINAL_EVALUATION_RESULTS_DIRECTORY)

    C_ENTROPY_RESULTS_DIRECTORY = os.path.join(f"Triple-CMR-query-{query.value}-dom-{dominating_modality.value}",f"c-entropy-results-{query.value}-query")
    TRIPLET_RESULTS_DIRECTORY = os.path.join(f"Triple-CMR-query-{query.value}-dom-{dominating_modality.value}",f"triplet-results-{query.value}-query")

    # Load saved model paths 
    saved_c_entropy_model_path = f"{C_ENTROPY_RESULTS_DIRECTORY}/c-entropy-model.pth"
    saved_triplet_model_path_query_fused = f"{TRIPLET_RESULTS_DIRECTORY}/triplet_model_best.pth"

    # Device to cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CrossSensoryNetwork(query).to(device)
    model.load_state_dict(torch.load(saved_c_entropy_model_path))
    model.eval()

    # Initialize test dataloader
    batch_size = 5  
    dataloader = get_test_loader(batch_size)
    test_loader = dataloader['test']

    # Dictionaries to save embeddings
    audio_embeddings_test = defaultdict(list)
    tactile_embeddings_test = defaultdict(list)
    visual_embeddings_test = defaultdict(list)
    fused_embeddings_test = defaultdict(list)

    # Inference loop
    with torch.no_grad():
        for i, (audio_input, tactile_input, visual_input, targets) in enumerate(test_loader):
            audio_input, tactile_input, visual_input, targets = audio_input.to(device), tactile_input.to(device), visual_input.to(device), targets.to(device)

            # Get outputs and embeddings
            audio_output, tactile_output, visual_output, attention_out, _ = model(audio_input, tactile_input, visual_input)

            for j in range(audio_output.shape[0]):
                label = targets[j].item()
                audio_embeddings_test[label].append(audio_output[j].cpu().numpy())
                tactile_embeddings_test[label].append(tactile_output[j].cpu().numpy())
                visual_embeddings_test[label].append(visual_output[j].cpu().numpy())
                fused_embeddings_test[label].append(attention_out[j].cpu().numpy())

    # Save the intermediate state of embeddings
    np.save(os.path.join(f"{FINAL_EVALUATION_RESULTS_DIRECTORY}","audio_embeddings_test_c_entropy.npy"), dict(audio_embeddings_test))
    np.save(os.path.join(f"{FINAL_EVALUATION_RESULTS_DIRECTORY}","tactile_embeddings_test_c_entropy.npy"), dict(tactile_embeddings_test))
    np.save(os.path.join(f"{FINAL_EVALUATION_RESULTS_DIRECTORY}","visual_embeddings_test_c_entropy.npy"), dict(visual_embeddings_test))
    np.save(os.path.join(f"{FINAL_EVALUATION_RESULTS_DIRECTORY}","fused_embeddings_test_c_entropy.npy"), dict(fused_embeddings_test))


    # Initialize triplet model
    model = EmbeddingNet(embedding_dim=200).to(device)
    model.load_state_dict(torch.load(saved_triplet_model_path_query_fused))
    model.eval()

    # Produce final query embeddings from test set
    query_embeddings = np.load(os.path.join(FINAL_EVALUATION_RESULTS_DIRECTORY, f"{query.value}_embeddings_test_c_entropy.npy"), allow_pickle=True).item()
    fused_embeddings = np.load(os.path.join(FINAL_EVALUATION_RESULTS_DIRECTORY, "fused_embeddings_test_c_entropy.npy"), allow_pickle=True).item()
    
    with torch.no_grad():
        new_query_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in query_embeddings.items()}
        new_fused_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in fused_embeddings.items()}

    # Load retrieval space embeddings
    retrieval_query_embeddings = np.load(os.path.join(C_ENTROPY_RESULTS_DIRECTORY, f"{query.value}_embeddings_train.npy"), allow_pickle=True).item()
    retrieval_fused_emebddings = np.load(os.path.join(C_ENTROPY_RESULTS_DIRECTORY, f"fused_embeddings_train.npy"), allow_pickle=True).item()

    with torch.no_grad():
        new_retrieval_query_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in retrieval_query_embeddings.items()}
        new_retrieval_fused_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in retrieval_fused_emebddings.items()}
    
    # Evaluate MAP score
    MAP_fused2query, MAP_query2fused = evaluate(new_query_embeddings, new_fused_embeddings, new_retrieval_query_embeddings, new_retrieval_fused_embeddings)

    logger.log("Finished evaluation of final performance.")
    logger.log(f"MAP Query Modality to Fused: {MAP_query2fused}")
    logger.log(f"MAP Fused Modality to Query: {MAP_fused2query}")

    RESULTS_DIRECTORY = f'Triple-CMR-query-{query.value}-dom-{dominating_modality.value}'

    with open(f"{RESULTS_DIRECTORY}/result.txt", "w") as file:
        file.write(f"MAP Query Modality to Fused: {MAP_query2fused}")
        file.write(f"MAP Fused to Initial Query: {MAP_fused2query}")
        