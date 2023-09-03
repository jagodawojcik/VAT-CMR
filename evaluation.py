import numpy as np

RANK_K = 5

def prepare_data_for_evaluation(embeddings):
    labels = []
    embeddings_list = []
    for label, embedding in embeddings.items():
        labels.extend([label]*len(embedding))
        embeddings_list.extend(embedding)
    return np.array(embeddings_list), np.array(labels)

def average_precision(relevant_scores, k):
    if len(relevant_scores) == 0:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, relevant in enumerate(relevant_scores[:k]):
        if relevant > 0:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(relevant_scores), k)

def calculate_MAP(queries, retrievals, query_labels, retrieval_labels):
    APs = []
    for i in range(len(queries)):
        query = queries[i]
        current_query_label = query_labels[i]

        # calculate the distance between the query and all retrievals
        distances = np.linalg.norm(retrievals - query, axis=1)

        # sort the distances to find the closest and get their indices
        closest_indices = np.argsort(distances)

        # get the labels for the closest instances
        closest_labels = retrieval_labels[closest_indices]

        # determine which retrievals are true positives
        relevant_retrievals = (closest_labels == current_query_label)

        # calculate the average precision for this query
        ap = average_precision(relevant_retrievals, RANK_K)
        APs.append(ap)

    # the mean average precision is the mean of the average precision values for all queries
    return np.mean(APs)

def evaluate(query_test_embeddings, fused_test_embeddings, query_train_embeddings, fused_train_embeddings):
    query_test_queries, query_test_query_labels = prepare_data_for_evaluation(query_test_embeddings)
    fused_test_queries, fused_test_query_labels = prepare_data_for_evaluation(fused_test_embeddings)
    query_train_queries, query_train_query_labels = prepare_data_for_evaluation(query_train_embeddings)
    fused_train_queries, fused_train_query_labels = prepare_data_for_evaluation(fused_train_embeddings)

    # Calculate MAP for tactile2visual retrieval
    MAP_fused2query = calculate_MAP(fused_test_queries, query_train_queries, fused_test_query_labels, query_train_query_labels)

    # Calculate MAP for visual2tactile retrieval
    MAP_query2fused = calculate_MAP(query_test_queries, fused_train_queries, query_test_query_labels, fused_train_query_labels)

    return MAP_fused2query, MAP_query2fused
