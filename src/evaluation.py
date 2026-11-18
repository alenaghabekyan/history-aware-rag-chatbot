
import math

from src.retrival_pipeline import retrieve_documents
from src.evaluation_dataset import evaluation_dataset


# ==========================================
# 1. RELEVANCE CHECK
# ==========================================

def is_relevant(doc, relevant_keywords):
    """
    Check whether a document contains all expected keywords.
    """
    text = doc.page_content.lower()

    return all(
        keyword.lower() in text
        for keyword in relevant_keywords
    )


# ==========================================
# 2. PRECISION@K
# ==========================================

def calculate_precision_at_k(
    retrieved_docs,
    relevant_keywords,
    k=5
):
    """
    Calculate Precision@K.
    Missing retrieved documents count as non-relevant.
    """

    top_k_docs = retrieved_docs[:k]

    if not top_k_docs:
        return 0.0

    relevant_count = sum(
        is_relevant(doc, relevant_keywords)
        for doc in top_k_docs
    )

    return relevant_count / k


# ==========================================
# 3. DCG@K
# ==========================================

def calculate_dcg(relevance_scores, k=5):
    """
    Calculate Discounted Cumulative Gain.
    """

    scores = relevance_scores[:k]

    dcg = 0.0

    for i, relevance in enumerate(scores):
        position = i + 1

        dcg += (
            (2 ** relevance - 1)
            / math.log2(position + 1)
        )

    return dcg


# ==========================================
# 4. NDCG@K
# ==========================================

def calculate_ndcg(relevance_scores, k=5):
    """
    Calculate Normalized Discounted Cumulative Gain.
    """

    dcg = calculate_dcg(relevance_scores, k)

    ideal_scores = sorted(
        relevance_scores,
        reverse=True
    )

    idcg = calculate_dcg(ideal_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ==========================================
# 5. MAIN EVALUATION
# ==========================================

def evaluate_retrieval():

    total_questions = len(evaluation_dataset)

    if total_questions == 0:
        print("Evaluation dataset is empty.")
        return

    k = 5

    successful_retrievals = 0

    reciprocal_ranks = []
    precision_scores = []
    ndcg_scores = []

    for item in evaluation_dataset:

        question = item["question"]

        expected_keywords = item["expected_keywords"]

        relevant_keywords = item["relevant_keywords"]

        # Retrieve documents
        retrieved_docs = retrieve_documents(question)

        # ----------------------------------
        # RECALL@5
        # ----------------------------------

        combined_text = " ".join(
            doc.page_content.lower()
            for doc in retrieved_docs[:k]
        )

        found = all(
            keyword.lower() in combined_text
            for keyword in expected_keywords
        )

        if found:
            successful_retrievals += 1

        # ----------------------------------
        # MRR
        # ----------------------------------

        relevant_rank = None

        for rank, doc in enumerate(retrieved_docs[:k], start=1):

            if is_relevant(doc, relevant_keywords):

                relevant_rank = rank
                break

        if relevant_rank is not None:
            reciprocal_rank = 1 / relevant_rank
        else:
            reciprocal_rank = 0

        reciprocal_ranks.append(reciprocal_rank)

        # ----------------------------------
        # PRECISION@5
        # ----------------------------------

        precision = calculate_precision_at_k(
            retrieved_docs,
            relevant_keywords,
            k
        )

        precision_scores.append(precision)

        # ----------------------------------
        # NDCG@5
        # ----------------------------------

        # Temporary binary relevance labels:
        # 1 = relevant
        # 0 = not relevant

        relevance_scores = [
            1 if is_relevant(doc, relevant_keywords) else 0
            for doc in retrieved_docs[:k]
        ]

        # Pad missing documents with zero relevance
        relevance_scores += [0] * (k - len(relevance_scores))

        ndcg = calculate_ndcg(
            relevance_scores,
            k
        )

        ndcg_scores.append(ndcg)

        # ----------------------------------
        # PRINT QUESTION RESULTS
        # ----------------------------------

        print("\nQuestion:", question)
        print("Retrieved documents:", len(retrieved_docs))
        print("Relevant information found:", found)
        print("Relevant document rank:", relevant_rank)

        print(f"Reciprocal Rank: {reciprocal_rank:.2f}")
        print(f"Precision@{k}: {precision:.2f}")
        print(f"NDCG@{k}: {ndcg:.4f}")

    # ======================================
    # FINAL METRICS
    # ======================================

    recall_at_k = successful_retrievals / total_questions

    mrr = sum(reciprocal_ranks) / total_questions

    mean_precision = sum(precision_scores) / total_questions

    mean_ndcg = sum(ndcg_scores) / total_questions

    print("\n========== EVALUATION RESULTS ==========")

    print(f"Total questions: {total_questions}")

    print(f"Successful retrievals: {successful_retrievals}")

    print(f"Recall@{k}: {recall_at_k:.2%}")

    print(f"MRR: {mrr:.4f}")

    print(f"Precision@{k}: {mean_precision:.2%}")

    print(f"NDCG@{k}: {mean_ndcg:.4f}")

    print("========================================")


if __name__ == "__main__":
    evaluate_retrieval()
