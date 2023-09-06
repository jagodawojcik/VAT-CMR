from train_c_entropy import train_with_cross_entropy
from train_triplet_loss import train_with_triplet_loss
from performance_eval import test_set_performance_evaluate
from logger import logger
from modality import QueryModality, DominatingModality
import warnings
import argparse
import os


# Default Training Parameters
EPOCHS_PRETRAIN = 15
EPOCHS_C_ENTROPY = 50
BATCH_SIZE_C_ENTROPY = 5
EPOCHS_TRIPLET_LOSS = 20001
BATCH_SIZE_TRIPLET_LOSS = 1
MARGIN = 0.5


def main(args):
    # Ignore user type of warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    RESULTS_DIRECTORY = f'Triple-CMR-query-{args.query_modality.value}-dom-{args.dominating_modality.value}'

    #Create a directory to save your results
    if os.path.exists(RESULTS_DIRECTORY): 
        raise Exception(f"Directory {RESULTS_DIRECTORY} already exists, please check for existing results.")

    logger.log(f"Directory {RESULTS_DIRECTORY} does not exist, creating...")
    os.makedirs(RESULTS_DIRECTORY)

    with open(f"{RESULTS_DIRECTORY}/information.txt", "w") as file:
        file.write(f"Training Triple-CMR Model.")
        file.write(f"\nQuery Modality: {args.query_modality.value}")
        file.write(f"\nClassifiaction with {args.dominating_modality.value} dominating modality.")

    logger.log("---------Starting Cross Entropy Training-----------")
    train_with_cross_entropy(query=args.query_modality, dominating_modality=args.dominating_modality, epochs_pre=args.epoch_pretrain, epochs_c_entropy=args.epoch_c_entropy, batch_size=args.batch_size_c_entropy)
    logger.log("-----------Cross Entropy Training Completed-----------")

    logger.log("----------Starting Triplet Loss Training-----------")
    train_with_triplet_loss(query=args.query_modality, dominating=args.dominating_modality, epochs=args.epoch_triplet, batch_size=args.batch_size_triplet, margin=args.margin_triplet)
    logger.log("----------Triplet Loss Training Completed-----------")

    logger.log(("-----------Start Final Performance Evaluation-----------"))
    test_set_performance_evaluate(query=args.query_modality, dominating_modality=args.dominating_modality)
    logger.log(("-----------Performance Evaluation Completed-----------"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process different modes and set parameters')

    # Define the command-line options
    parser.add_argument('--query-modality', choices=[e.value for e in QueryModality], required=True, help='Set the query modality')
    parser.add_argument('--dominating-modality', choices=[e.value for e in DominatingModality], required=True, help='Set the dominating modality')
    parser.add_argument('--epoch-pretrain', default=EPOCHS_PRETRAIN, help='Set number of epochs for tactile branch pretraining with cross-entropy.')
    parser.add_argument('--epoch-c-entropy', default=EPOCHS_C_ENTROPY, help='Set number of epochs for cross-entropy stage')
    parser.add_argument('--batch-size-c-entropy', default=BATCH_SIZE_C_ENTROPY, help='Set batch size for cross-entropy training stage.')
    parser.add_argument('--epoch-triplet', default=EPOCHS_TRIPLET_LOSS, help='Set max number of epochs for triplet loss training')
    parser.add_argument('--batch-size-triplet', default=BATCH_SIZE_TRIPLET_LOSS, help='Set batch size for triplet loss training stage.')
    parser.add_argument('--margin-triplet', default=BATCH_SIZE_TRIPLET_LOSS, help='Set margin size for triplet loss optimization.')
    parser.add_argument('--use-linux-echo', default=False, help='Set logging type as True for linux echo print or False for standard Python print.')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    args.query_modality = QueryModality(args.query_modality)
    args.dominating_modality = DominatingModality(args.dominating_modality)
    # Convert integer arguments
    args.epoch_pretrain = int(args.epoch_pretrain)
    args.epoch_c_entropy = int(args.epoch_c_entropy)
    args.batch_size_c_entropy = int(args.batch_size_c_entropy)
    args.epoch_triplet = int(args.epoch_triplet)
    args.batch_size_triplet = int(args.batch_size_triplet)
    args.margin_triplet = float(args.margin_triplet)

    # Set logger type
    logger.use_echo = args.use_linux_echo

    # Call the main function with parsed arguments
    main(args) 
    