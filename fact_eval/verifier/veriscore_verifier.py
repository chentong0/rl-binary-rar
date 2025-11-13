import os
import json

from vllm import LLM
from transformers import AutoTokenizer
import torch
import gc
from tqdm import tqdm
from fact_eval.prompts.prompt_veriscore_verifier import get_veriscore_verifier_message_ft, get_veriscore_verifier_prompt_gpt
import os
from fact_eval.utils.async_completion import batch_chat_complete

class ClaimVerifier():
    def __init__(self, model_name, lazy_loading=True):
        # self.model = None
        self.llm = None
        self.client = None
        self.model_name = model_name
        self.lazy_loading = lazy_loading
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def load_model(self):
        from fact_eval.utils.load_model import load_model
        model_info = load_model(self.model_name)
        self.client = model_info["client"]
        self.tokenizer = model_info["tokenizer"]
        self.llm = model_info["llm"]

    def unload_model(self):
        if self.llm:
            del self.llm
            del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        elif self.client:
            del self.client
            gc.collect()

    def batch_verifying_claim(self, claim_snippets_dict, search_res_num=5):
        """
        search_snippet_lst = [{"title": title, "snippet": snippet, "link": link}, ...]
        """

        if self.lazy_loading:
            self.load_model()

        prompts = []
        results = {}
        for claim, search_snippet_lst in claim_snippets_dict.items():
            search_res_str = ""
            search_cnt = 1
            for search_dict in search_snippet_lst[:search_res_num]:
                search_res_str += f'Search result {search_cnt}\nTitle: {search_dict["title"].strip()}\nLink: {search_dict["link"].strip()}\nContent: {search_dict["snippet"].strip()}\n\n'
                search_cnt += 1
                
            usr_input = f"Claim: {claim.strip()}\n\n{search_res_str.strip()}"

            prompts.append(usr_input)

        if self.llm:
            for i, usr_input in enumerate(prompts):
                # prompt = self.alpaca_prompt.format(self.instruction, usr_input)
                # prompt = self.get_prompt(usr_input)
                message = get_veriscore_verifier_message_ft(usr_input)
                # prompt = self.apply_chat_template(message)
                prompt = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

                if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
                    prompt.removeprefix(self.tokenizer.bos_token)
                prompts[i] = prompt

            from vllm import SamplingParams
            outputs = self.llm.generate(
                prompts, sampling_params=SamplingParams(max_tokens=16, temperature=0)
            )
            for i, (claim, output) in enumerate(zip(claim_snippets_dict.keys(), outputs)):
                response = output.outputs[0].text
                clean_output = response.strip()
                results[claim] = clean_output == "supported"
        else:
            # outputs = []
            # completion_tokens = 0
            # prompt_tokens = 0
            # MAX_TRIES = 3
            # progress_bar = tqdm(total=len(prompts), desc="Processing Prompts", unit="prompt")
            # for prompt in prompts:
            #     for tries in range(MAX_TRIES):
            #         try:
            #             response = self.client.chat.completions.create(
            #                 model=self.model_name,
            #                 messages=[
            #                     # {"role": "system", "content": self.instruction},
            #                     {"role": "user", "content": get_veriscore_verifier_prompt_gpt(prompt)}
            #                 ],
            #                 max_tokens=16,
            #                 temperature=0
            #             )
            #             outputs.append(response)
            #             completion_tokens += response.usage.completion_tokens
            #             prompt_tokens += response.usage.prompt_tokens
            #             progress_bar.update(1)
            #             progress_bar.set_postfix({
            #                 "Completion Tokens": completion_tokens,
            #                 "Prompt Tokens": prompt_tokens,
            #             })
            #             break
            #         except Exception as e:
            #             print(f"Error: {e}. Retrying {tries + 1}/{MAX_TRIES}...")
            #     else:
            #         print(f"Failed to generate response for prompt: {prompt} after {MAX_TRIES} tries.")
            #         outputs.append(None)

            #     # print("-" * 20, prompt, "-" * 20)
            #     # print("-" * 20, response.choices[0].message.content, "-" * 20)
            #     # print(response)
            # for i, (claim, output) in enumerate(zip(claim_snippets_dict.keys(), outputs)):
            #     if output is not None:
            #         response = output.choices[0].message.content
            #         clean_output = response.strip()
            #         print(clean_output)
            #         results[claim] = clean_output == "supported"
            #     else:
            #         results[claim] = False
            messages_list = [[{"role": "user", "content": get_veriscore_verifier_prompt_gpt(prompt)}] for prompt in prompts]
            outputs = batch_chat_complete(
                self.client,
                messages_list,
                model=self.model_name.split("::")[-1],
                max_tokens=16,
                temperature=0
            )
            for i, (claim, output) in enumerate(zip(claim_snippets_dict.keys(), outputs)):
                try:
                    clean_output = output.choices[0].message.content.strip()
                    results[claim] = clean_output == "supported"
                    self.prompt_tokens += output.usage.prompt_tokens
                    self.completion_tokens += output.usage.completion_tokens
                except Exception as e:
                    print(f"Error: {e}")
                    results[claim] = False

        # json.dump({claim: {"prompt": output.prompt, "response": output.outputs[0].text} for claim, output in zip(claim_snippets_dict.keys(), outputs)}, open("debug.json", "w"), indent=4)
        
        if self.lazy_loading:
            self.unload_model()

        return results
