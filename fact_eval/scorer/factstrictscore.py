import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
# Initialize components
from fact_eval.search.document_collection_search import \
    SearchEngineDocumentCollection
from fact_eval.verifier.customized_strict_verifier import OpenaiStrictVerifier
from tqdm import tqdm

"""
Factuality Scorer Module

This module provides classes for scoring the factuality of language model responses.
It supports different strategies for decomposition, verification, and aggregation of factuality scores.
"""


@dataclass
class FactStrictScoreConfig:
    model_name_verification: Optional[str] = None
    prompt_type: str = "binary" # "binary" or "rating"
    search_model_name: str = "bm25-only"
    search_tokenizer_name: Optional[str] = None
    search_chunk_size: int = 250
    search_num_chunks: int = 16
    search_num_processes: int = 1
    cache_dir: Optional[str] = None
    # output_dir: Optional[str] = None
    batch_size: int = 256


class FactStrictScorer:
    """
    A class to compute FactScore for model responses by extracting claims,
    searching for evidence, and verifying claims against the evidence.
    """
    
    def __init__(self, config: FactStrictScoreConfig):
        """
        Initialize the FactScorer.
        
        Args:
            config: Configuration object containing all parameters
        """
        self.config = config
        self.cache_dir = config.cache_dir

        self.search_engine = SearchEngineDocumentCollection(
            chunk_size=config.search_chunk_size,
            tokenizer_name=config.search_tokenizer_name
        )
        self.verifier = OpenaiStrictVerifier(
            model_name=config.model_name_verification,
            prompt_type=config.prompt_type,
            batch_size=config.batch_size
        )

        # self.output_dir = config.output_dir
        # self.cache_dir = config.cache_dir

        # # Create directories if they don't exist
        # # if self.output_dir is not None:
        # #     os.makedirs(self.output_dir, exist_ok=True)
        # if self.cache_dir is not None:
        #     os.makedirs(self.cache_dir, exist_ok=True)


        # self.search_engine = SearchEngineDocumentCollection(
        #     chunk_size=config.search_chunk_size
        # )
        
        # self.claim_verifier = OpenaiClaimVerifier(model_name=config.model_name_verification, batch_size=config.batch_size)





    # def _extract_claims(self, data: List[Dict[str, Any]], model_alias: str) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
    #     """
    #     Extract claims from model responses.
        
    #     Args:
    #         data: List of data items containing model responses
    #         model_alias: Tag for saving intermediate results
            
    #     Returns:
    #         Tuple of (processed_data, extracted_claims) where extracted_claims is a list of lists
    #     """
    #     extracted_claims = []
        
    #     # Check if cache exists
    #     if self.cache_dir and os.path.exists(os.path.join(self.cache_dir, f"extractor_cache_{model_alias}.json")):
    #         with open(os.path.join(self.cache_dir, f"extractor_cache_{model_alias}.json"), "r") as f:
    #             cached_data = json.load(f)
    #         claims_list = cached_data["claims_list"]
    #         print(f"Loaded cached claims from {model_alias}")
    #     else:
    #         # Extract claims from prompts and responses
    #         prompt_list = [item["prompt"] for item in data]
    #         response_list = [item["response"] for item in data]
            
    #         claims_list = self.claim_extractor.extract(prompt_list, response_list)
            
    #         # Cache the results
    #         if self.cache_dir is not None:
    #             cache_file = os.path.join(self.cache_dir, f"extractor_cache_{model_alias}.json")
    #             with open(cache_file, "w") as f:
    #                 json.dump({
    #                     "claims_list": claims_list,
    #                     "prompt_list": prompt_list,
    #                     "response_list": response_list
    #                 }, f, indent=2)
    #             print(f"Claim extraction completed! Saved to {cache_file}")

    #     # Process data and collect claims as list of lists
    #     for i, (dict_item, claims) in enumerate(tqdm(zip(data, claims_list), desc="Processing claims")):
    #         data[i] = {
    #             **dict_item,
    #             "claim_list": claims
    #         }
    #         extracted_claims.append(claims)
        
    #     return data, extracted_claims

    def _search_evidence(self, prompt_list: List[str], response_list: List[str], docs_list: List[List[Dict[str, str]]], model_alias: str) -> Dict[str, List[str]]:
        """
        Search for evidence to support the given queries.

        Args:
            prompt_list: List of prompts to search evidence for
            response_list: List of responses to search evidence for
            docs_list: List of document collections for each data item (each doc has title and text)
            model_alias: Tag for saving intermediate results
        """

        if self.cache_dir and os.path.exists(os.path.join(self.cache_dir, f"search_cache_{model_alias}.json")):
            with open(os.path.join(self.cache_dir, f"search_cache_{model_alias}.json"), "r") as f:
                chunks_list = json.load(f)
            print(f"Loaded cached search results from {model_alias}")
            return chunks_list
        
        else:

            query_list = [f"{response_list[i]}" for i in range(len(prompt_list))]
            documents_list = [docs_list[i] for i in range(len(docs_list))]

            chunks_list = self.search_engine.search(
                documents_list=documents_list,
                query_list=query_list,
                k=self.config.search_num_chunks,
                num_processes=self.config.search_num_processes,
            )

            if self.cache_dir is not None:
                cache_file = os.path.join(self.cache_dir, f"search_cache_{model_alias}.json")
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(chunks_list, f, indent=2)
                print(f"Search completed! Saved to {cache_file}")

        return chunks_list


    # def _search_evidence(self, extracted_claims: List[List[str]], docs_list: List[List[Dict[str, str]]], model_alias: str) -> Dict[str, List[str]]:
    #     """
    #     Search for evidence to support the extracted claims.
        
    #     Args:
    #         extracted_claims: List of lists of claims to search evidence for
    #         docs_list: List of document collections for each data item (each doc has title and text)
    #         model_alias: Tag for saving intermediate results
            
    #     Returns:
    #         Dictionary mapping claims to search results
    #     """
    #     # Check if cache exists
    #     if self.cache_dir and os.path.exists(os.path.join(self.cache_dir, f"search_cache_{model_alias}.json")):
    #         with open(os.path.join(self.cache_dir, f"search_cache_{model_alias}.json"), "r") as f:
    #             claim_to_chunks = json.load(f)
    #         print(f"Loaded cached search results from {model_alias}")
    #     else:
    #         # Flatten claims and create mapping from claims to documents
    #         all_claims = []
    #         claim_to_docs = {}
            
    #         for claims_per_item, docs_per_item in zip(extracted_claims, docs_list):
    #             for claim in claims_per_item:
    #                 all_claims.append(claim)
    #                 claim_to_docs[claim] = docs_per_item

    #         # Create documents_list and query_list for search
    #         documents_list = [claim_to_docs[claim] for claim in all_claims]
    #         query_list = all_claims
            
    #         chunks_list = self.search_engine.search(
    #             documents_list=documents_list,
    #             query_list=query_list,
    #             k=self.config.search_num_chunks,
    #             num_processes=self.config.search_num_processes
    #         )
    #         claim_to_chunks = {claim: chunks for claim, chunks in zip(all_claims, chunks_list)}

    #         # Cache the results
    #         if self.cache_dir is not None:
    #             cache_file = os.path.join(self.cache_dir, f"search_cache_{model_alias}.json")
    #             with open(cache_file, "w") as f:
    #                 json.dump(claim_to_chunks, f, indent=2)
    #             print(f"Search completed! Saved to {cache_file}")

    #     return claim_to_chunks

    # def _verify_claims(self, claim_to_chunks: Dict[str, List[str]], model_alias: str) -> Dict[str, bool]:
    #     """
    #     Verify claims against the search results.
        
    #     Args:
    #         claim_to_chunks: Dictionary mapping claims to search results
    #         model_alias: Tag for saving intermediate results
            
    #     Returns:
    #         Dictionary mapping claims to verification results (True/False)
    #     """
    #     # Check if cache exists
    #     if self.cache_dir and os.path.exists(os.path.join(self.cache_dir, f"verification_cache_{model_alias}.json")):
    #         with open(os.path.join(self.cache_dir, f"verification_cache_{model_alias}.json"), "r") as f:
    #             claim_to_correctness = json.load(f)
    #         print(f"Loaded cached verification results from {model_alias}")
    #     else:
    #         # Extract claims and passages for verification
    #         all_claims = list(claim_to_chunks.keys())
    #         passages_list = [claim_to_chunks[claim] for claim in all_claims]
            
    #         # Verify claims
    #         correctness_list = self.claim_verifier.verify(
    #             claim_list=all_claims,
    #             passages_list=passages_list,
    #         )
    #         claim_to_correctness = {claim: correctness for claim, correctness in zip(all_claims, correctness_list)}

    #         # Cache the results
    #         if self.cache_dir is not None:
    #             cache_file = os.path.join(self.cache_dir, f"verification_cache_{model_alias}.json")
    #             with open(cache_file, "w") as f:
    #                 json.dump(claim_to_correctness, f, indent=2)
    #             print(f"Verification completed! Saved to {cache_file}")

    #     return claim_to_correctness

    def _verify_correctness(self, prompt_list: List[str], response_list: List[str], chunks_list: List[List[str]], model_alias: str) -> Dict[str, bool]:
        """
        Verify the correctness of the given queries.
        """

        if self.cache_dir and os.path.exists(os.path.join(self.cache_dir, f"verification_cache_{model_alias}.json")):
            with open(os.path.join(self.cache_dir, f"verification_cache_{model_alias}.json"), "r") as f:
                results_list = json.load(f)
            print(f"Loaded cached verification results from {model_alias}")
            return results_list
        
        else:

            results_list = self.verifier.verify(
                prompt_list=prompt_list,
                response_list=response_list,
                passages_list=chunks_list,
            )

            if self.cache_dir is not None:
                cache_file = os.path.join(self.cache_dir, f"verification_cache_{model_alias}.json")
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(results_list, f, indent=2)
                print(f"Verification completed! Saved to {cache_file}")

        return results_list
        

    # def _compute_metrics(self, data: List[Dict[str, Any]], claim_to_correctness: Dict[str, bool]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    #     """
    #     Compute FactScore metrics and per-instance results.
        
    #     Args:
    #         data: Processed data with claims
    #         claim_to_correctness: Verification results for claims
            
    #     Returns:
    #         Tuple of (aggregate_metrics, per_instance_results)
    #     """

    #     # Create per-instance results
    #     per_instance_results = []
    #     for dict_item in data:
    #         claim_list = dict_item.get("claim_list", [])
    #         factuality_score = None
    #         correct_claims = sum(1 for claim in claim_list if claim_to_correctness.get(claim, False))
    #         num_claims = len(claim_list)
    #         if len(claim_list) > 0:
    #             factuality_score = correct_claims / len(claim_list)
    #             penalty = 1.0 if len(claim_list) > self.config.length_penalty_threshold else np.exp(1 - self.config.length_penalty_threshold / len(claim_list))
    #             factuality_score_with_length_penalty = factuality_score * penalty
    #         else:
    #             factuality_score = None
    #             penalty = 0.0
    #             factuality_score_with_length_penalty = 0.0

    #         per_instance_results.append({
    #             **{k: v for k, v in dict_item.items() if k not in ["docs"]},
    #             "factuality_score": factuality_score,
    #             "factuality_score_with_length_penalty": factuality_score_with_length_penalty,
    #             "num_correct_claims": correct_claims,
    #             "num_claims": num_claims,
    #         })

    #     # Compute aggregate metrics
    #     aggregate_metrics = self._aggregate_metrics(per_instance_results)
        
    #     return aggregate_metrics, per_instance_results

    # def _aggregate_metrics(self, per_instance_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     """
    #     Compute aggregate FactScore metrics from per-instance results.
    #     """
    #     # Group by domain and model for metric computation
    #     domain_to_subset = defaultdict(list)
        
    #     for dict_item in per_instance_results:
    #         domain = dict_item.get('source', 'unknown')
    #         model_name = dict_item.get('model', 'unknown')
    #         key = f"{domain}/{model_name}"
    #         domain_to_subset[key].append(dict_item)

    #     results = {}
    #     for key, subset in domain_to_subset.items():
    #         if any(item.get("factuality_score") is not None for item in subset):
    #             results[key] = {
    #                 "factuality_score": np.mean([item.get("factuality_score") for item in subset if item.get("factuality_score") is not None]),
    #                 "factuality_score_with_length_penalty": np.mean([item.get("factuality_score_with_length_penalty") for item in subset]),
    #                 "num_correct_claims": np.mean([item.get("num_correct_claims") for item in subset]),
    #                 "num_claims": np.mean([item.get("num_claims") for item in subset]),
    #                 "abstain_ratio": np.mean([1 if item.get("factuality_score") is None else 0 for item in subset]),
    #             }
    #         else:
    #             results[key] = {
    #                 "factuality_score": None,
    #                 "factuality_score_with_length_penalty": None,
    #                 "num_correct_claims": None,
    #                 "num_claims": None,
    #                 "abstain_ratio": None,
    #             }
    #     return results


    # def _get_factscore_metrics(self, model_domain_triplet_dict: Dict[str, Dict[str, List[List[int]]]]) -> Dict[str, Any]:
    #     """
    #     Compute aggregate FactScore metrics from triplets.
        
    #     Args:
    #         model_domain_triplet_dict: Dictionary mapping domain -> model -> list of triplets
            
    #     Returns:
    #         Dictionary containing aggregate metrics
    #     """
    #     metrics = {}
        
    #     for domain, model_dict in model_domain_triplet_dict.items():
    #         for model_name, triplets in model_dict.items():
    #             total_correct = sum(triplet[0] for triplet in triplets)
    #             total_claims = sum(triplet[1] for triplet in triplets)
                
    #             precision = total_correct / total_claims if total_claims > 0 else 0.0
                
    #             key = f"{domain}/{model_name}"
    #             metrics[key] = {
    #                 "precision": precision,
    #                 "total_correct_claims": total_correct,
    #                 "total_claims": total_claims,
    #                 "num_instances": len(triplets)
    #             }
        
    #     return metrics

    def _compute_metrics(self, data: List[Dict[str, Any]], results_list: List[bool]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Compute metrics from the correctness list.
        """
        # Create per-instance results
        per_instance_results = []
        for i, (dict_item, result_item) in enumerate(zip(data, results_list)):
            if self.config.prompt_type == "rating":
                dict_item["factuality_score"] = result_item["SCORE"] / 10.0
                dict_item["factuality_score_explanation"] = result_item["REASONING"]
            elif self.config.prompt_type == "binary":
                dict_item["factuality_score"] = result_item["SCORE"]
                dict_item["factuality_score_explanation"] = result_item["REASONING"]

            per_instance_results.append({
                **{k: v for k, v in dict_item.items() if k not in ["docs"]},
                "factuality_score": dict_item["factuality_score"],
                "factuality_score_explanation": dict_item["factuality_score_explanation"],
            })
        
        aggregate_metrics = {}
        return aggregate_metrics, per_instance_results


    def get_score(self, data: List[Dict[str, Any]], model_alias: str = 'default') -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        """

        if not data:
            raise ValueError("Data list cannot be empty")
        if not model_alias:
            raise ValueError("model_alias cannot be empty")
        if not all(isinstance(item.get("prompt", None), str) for item in data):
            raise ValueError("prompt is required for all data items")
        if not all(isinstance(item.get("response", None), str) for item in data):
            raise ValueError("response is required for all data items")
        if not all(isinstance(item.get("docs", None), list) for item in data):
            raise ValueError("docs is required for all data items")

        # first retrieve
        query_list = [item["prompt"] for item in data]
        response_list = [item["response"] for item in data]
        docs_list = [item["docs"] for item in data]

        # Step 1: Retrieve evidence
        chunks_list = self._search_evidence(query_list, response_list, docs_list, model_alias)

        # Step 2: Verify claims
        results_list = self._verify_correctness(query_list, response_list, chunks_list, model_alias)


        # Step 3: Compute metrics
        aggregate_metrics, per_instance_results = self._compute_metrics(data, results_list)

        # if self.cache_dir is not none, save per_instance_results
        if self.cache_dir is not None:
            output_path = os.path.join(self.cache_dir, 'scores.json')
            with open(output_path, "w") as f:
                json.dump(per_instance_results, f, indent=2)
            print(f"Scores saved to {output_path}")

        return aggregate_metrics, per_instance_results

    # def get_score(self, data: List[Dict[str, Any]], model_alias: str = 'default') -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    #     """
    #     Compute FactScore for the given data.
        
    #     Args:
    #         data: List of data items containing model responses
    #         model_alias: Tag for saving intermediate and final results
            
    #     Returns:
    #         Tuple of (aggregate_metrics, per_instance_results)
    #     """
    #     if not data:
    #         raise ValueError("Data list cannot be empty")
            
    #     if not model_alias:
    #         raise ValueError("model_alias cannot be empty")

    #     # Extract docs_list from data
    #     docs_list = [item.get("docs", []) for item in data]
    #     if not any(docs_list):
    #         raise ValueError("docs_list is required for all data items")
    #     if not all(isinstance(item.get("prompt", None), str) for item in data):
    #         raise ValueError("prompt is required for all data items")
    #     if not all(isinstance(item.get("response", None), str) for item in data):
    #         raise ValueError("response is required for all data items")

    #     # Step 1: Extract claims
    #     data, extracted_claims = self._extract_claims(data, model_alias)

    #     # Step 2: Search for evidence
    #     claim_to_chunks = self._search_evidence(extracted_claims, docs_list, model_alias)

    #     # Step 3: Verify claims
    #     claim_to_correctness = self._verify_claims(claim_to_chunks, model_alias)

    #     # Step 4: Compute metrics
    #     aggregate_metrics, per_instance_results = self._compute_metrics(data, claim_to_correctness)

    #     # # Save final results
    #     # if self.output_dir is not None:
    #     #     output_dir = os.path.join(self.output_dir, 'results')
    #     #     output_path = os.path.join(output_dir, f"factscore_results_{model_alias}.json")
    #     #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     #     with open(output_path, "w") as f:
    #     #         json.dump(aggregate_metrics, f, indent=2)

    #     if self.cache_dir is not None:
    #         # save token usage
    #         with open(os.path.join(self.cache_dir, f"token_usage.json"), "w") as f:
    #             json.dump({
    #                 "extraction": {
    #                     "prompt_tokens": self.claim_extractor.prompt_tokens,
    #                     "completion_tokens": self.claim_extractor.completion_tokens,
    #                 },
    #                 "verification": {
    #                     "prompt_tokens": self.claim_verifier.prompt_tokens,
    #                     "completion_tokens": self.claim_verifier.completion_tokens,
    #                 }
    #             }, f, indent=2)

    #     return aggregate_metrics, per_instance_results
