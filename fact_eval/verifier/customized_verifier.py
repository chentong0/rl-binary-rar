import os
from fact_eval.utils.async_completion import batch_chat_complete
from fact_eval.prompts.prompt_customized_verifier import get_customized_verifier_prompt

class ClaimVerifier:
    def __init__(self, *args, **kwargs):
        pass

    def verify(self, claim):
        raise NotImplementedError

class OpenaiClaimVerifier:
    def __init__(self, model_name, batch_size=256, prompt_type="support"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.prompt_type = prompt_type
        
        # must be one of "openai::", "vllm-openai::", "azure::
        assert self.model_name.startswith("openai::") or self.model_name.startswith("vllm-openai::") or self.model_name.startswith("azure::")
        from fact_eval.utils.load_model import load_model
        model_info = load_model(self.model_name)
        self.client = model_info["client"]
        self.tokenizer = model_info["tokenizer"]
        self.llm = model_info["llm"]

        # self.token_usage = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_prompt(self, claim, passages):
        return get_customized_verifier_prompt(claim, passages)


    def verify(self, claim_list, passages_list):
        task_prompt_list = [
            self.get_prompt(claim, "\n".join([f"Doc [{i}] {p}" for i, p in enumerate(passages)])) 
            for claim, passages in zip(claim_list, passages_list)
        ]
        messages_list = [[{"role": "user", "content": prompt}] for prompt in task_prompt_list]
        chat_completion_kwargs = {}
        if self.model_name.startswith("vllm-openai::"):
            chat_completion_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        outputs = batch_chat_complete(
            self.client,
            messages_list, batch_size=self.batch_size,
            model=self.model_name.split("::")[-1],
            max_tokens=10,
            **chat_completion_kwargs
        )
        # return [result.choices[0].text for result in results]
        # inconclusive and contradicted -> false, supported -> true
        results = []
        for output in outputs:
            try:
                # Sum up token usage from all outputs
                # self.token_usage += output.usage.total_tokens
                self.prompt_tokens += output.usage.prompt_tokens
                self.completion_tokens += output.usage.completion_tokens
                if self.prompt_type == "support":
                    correctness = output.choices[0].message.content == "supported"
                elif self.prompt_type == "contradiction":
                    correctness = (output.choices[0].message.content == "supported" or output.choices[0].message.content == "inconclusive")
                else:
                    raise ValueError(f"Invalid prompt type: {self.prompt_type}")
                results.append(correctness)
            except:
                results.append(False)
        
        return results


# class FactScoreClaimVerifier:
#     pass

# class VeriScoreClaimVerifier:
#     pass

# class EfficientClaimVerifier:
#     def __init__(self, model_name, api_base=None, prompt_type=None, search_type=None):
#         # Modify OpenAI's API key and API base to use vLLM's API server.
#         self.model_name = model_name
#         from openai import OpenAI
#         self.client = OpenAI(
#             base_url=api_base,
#         )
#         self.prompt_type = prompt_type
#         self.search_type = search_type
#         # if prompt_type is fava, search_type cannot be None
#         assert not (prompt_type == "fava" and search_type is None), f"Prompt type {prompt_type} requires search type to be specified."

#         # completion = client.completions.create(model=model_name, prompt="San Francisco is a")
#         # print("Completion result:", completion)
    
#     @staticmethod
#     def apply_chat_template(messages, add_generation_prompt=True, bos_token="<|begin_of_text|>"):
#         from jinja2 import Template
#         llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
#         # Create a template object
#         template = Template(llama3_chat_template)
#         # Render the template with the messages object
#         formatted_output = template.render(messages=messages, bos_token=bos_token, add_generation_prompt=add_generation_prompt)
#         return formatted_output

#     def verify_no_context(self, text_list):
#         from verification_prompt import get_verification_prompt_no_context
#         verification_prompt_list = []
#         for text in text_list:
#             verification_prompt = get_verification_prompt_no_context(text)
#             verification_prompt_list.append(verification_prompt)

#         messages_list = [[{"role": "user", "content": verification_prompt}] for verification_prompt in verification_prompt_list]
#         verification_prompt_chat_list = [self.apply_chat_template(messages, bos_token="") for messages in messages_list]
#         # print(verification_prompt_chat_list)
#         from utils_vllm import batch_chat_complete, batch_complete
#         results = batch_complete(
#             self.client,
#             verification_prompt_chat_list,
#             model=self.model_name,
#             max_tokens=1,
#             temperature=0,
#             logprobs=4,
#         )
#         import math
#         confidence_scores = []
#         for response in results:
#             logprobs_dict = response.choices[0].logprobs.top_logprobs[0]
#             correct_logits, incorrect_logits = logprobs_dict.get("A", -math.inf), logprobs_dict.get("B", -math.inf)
#             # if both are inf then set correct to be zero
#             if correct_logits == -math.inf and incorrect_logits == -math.inf:
#                 correct_prob = 0.0
#                 incorrect_prob = 1.0
#                 print("[WARNING] Both logits are -inf, setting correct_prob to 0 and incorrect_prob to 1")
#             else:
#                 correct_prob = math.exp(correct_logits) / (math.exp(correct_logits) + math.exp(incorrect_logits))
#                 incorrect_prob = 1.0 - correct_prob
#             confidence_scores.append(correct_prob)
#         return confidence_scores, None

#     def verify_fava(self, text_list, query_list, docs_list):
#         from verification_prompt import get_verification_prompt_fava
#         import numpy as np
#         from rank_bm25 import BM25Okapi
#         from utils_vllm import batch_chat_complete, batch_complete
#         import re
#         chunk_size = 256
#         verification_prompt_list = []
#         for text, query, docs in zip(text_list, query_list, docs_list):
#             doc_text = "\n".join(docs)
#             # split the docs into chunks of 256 tokens
#             docs_processed = []
#             for doc in docs:
#                 tokenized_doc = doc.split()
#                 num_chunks = len(tokenized_doc) // chunk_size + 1
#                 tokenizerd_chunk_list = [tokenizerd_chunk for tokenizerd_chunk in np.array_split(tokenized_doc, num_chunks)]
#                 docs_processed.extend(tokenizerd_chunk_list)
#             # print(f"Number of chunks: {len(docs_processed)}. Len: {[len(doc) for doc in docs_processed]}")
#             # use bm25 to get the top 5 docs
#             bm25 = BM25Okapi(docs_processed)
#             tokenized_text = text.split()
#             # print(f"pass len: {len(tokenized_text)}")
#             doc_scores = bm25.get_scores(tokenized_text)
#             # print(f"BM25 scores: {doc_scores}")
#             # self.search_type -- top{k} -- extract the k
#             top_k = int(re.search(r"top(\d+)", self.search_type).group(1))
#             top_docs_idx = np.argsort(-doc_scores)[:top_k]
#             top_docs = [f"Snippet {i}\n" + " ".join(docs_processed[idx]) for i, idx in enumerate(top_docs_idx)]
#             # get the top 5 docs
#             doc_text = "\n\n".join(top_docs)
            
#             verification_prompt = get_verification_prompt_fava(text, query, doc_text)
#             verification_prompt_list.append(verification_prompt)

#         messages_list = [[{"role": "user", "content": verification_prompt}] for verification_prompt in verification_prompt_list]
#         verification_prompt_chat_list = [self.apply_chat_template(messages, bos_token="") for messages in messages_list]
        
#         # print(verification_prompt_chat_list)

#         results = batch_complete(
#             self.client,
#             verification_prompt_chat_list,
#             model=self.model_name,
#             max_tokens=1024,
#             temperature=0,
#         )
#         # if the response contains <entity> or <relation> or <contradictory> -- indicate that the model is not sure
#         scores = []
#         for response in results:
#             # print(response)
#             response_text = response.choices[0].text
#             if "<error>" in response_text:
#                 scores.append(0.0)
#             else:
#                 scores.append(1.0)
#         metainfo = [
#             {
#                 "input": messages_list[i],
#                 "output": results[i].choices[0].text,
#             }
#             for i in range(len(messages_list))
#         ]
#         return scores, metainfo

#     def verify(self, text_list, query_list=None, docs_list=None):
#         if self.prompt_type == "no-context":
#             return self.verify_no_context(text_list)
#         elif self.prompt_type == "fava":
#             return self.verify_fava(text_list, query_list=query_list, docs_list=docs_list)
#         raise NotImplementedError(f"Prompt type {self.prompt_type} not implemented")


# test case
if __name__ == "__main__":
    verifier = OpenaiClaimVerifier(model_name="gpt-4.1-mini-standard")
    claim_list = [
        "The Eiffel Tower was completed in 1889 and stands 324 meters tall.",
        "Mozart wrote his first symphony at age 8 while living in London.",
        "The human body has exactly 206 bones throughout adulthood.",
        "Climate change has caused global sea levels to rise by 8 inches since 1900."
    ]
    passages_list = [
        [
            "The Eiffel Tower was completed on March 31, 1889. Standing at 324 meters, it remained the world's tallest structure until 1930.",
            "Paris's iconic tower took just over 2 years to construct and weighs approximately 10,000 tons.",
            "The tower actually grows in height by up to 6 inches in summer due to thermal expansion."
        ],
        [
            "Mozart composed his first symphony, K. 16, at age 8 while in London in 1764.",
            "However, historical records show Mozart never visited London until he was 9 years old.",
            "His first symphony was actually written in 1764 while in Paris."
        ],
        [
            "The adult human skeleton is typically made up of 206 bones.",
            "However, some people may have extra bones, like cervical ribs or extra fingers.",
            "The number of bones can also decrease with age as some bones fuse together."
        ],
        [
            "According to NASA, global sea levels have risen about 8-9 inches since 1880.",
            "The rate of sea level rise has accelerated in recent decades due to climate change.",
            "Thermal expansion and melting ice sheets are the primary drivers of rising seas."
        ]
    ]
    results = verifier.verify(claim_list, passages_list)
    print(results)
