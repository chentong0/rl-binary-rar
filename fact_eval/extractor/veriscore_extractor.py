import spacy
from vllm import LLM
from transformers import AutoTokenizer
import torch
import gc
from tqdm import tqdm
from fact_eval.prompts.prompt_veriscore_extractor import get_veriscore_extractor_prompt_ft, get_veriscore_extractor_prompt_gpt

from fact_eval.utils.async_completion import batch_chat_complete

import os


class ClaimExtractor():
    def __init__(self, model_name, lazy_loading=True):
        self.llm = None
        self.client = None
        self.model_name = model_name
        self.lazy_loading = lazy_loading
        self.prompt_tokens = 0
        self.completion_tokens = 0

        try:
            self.spacy_nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            # install spacy model
            from spacy.cli.download import download
            download("en_core_web_sm")
            self.spacy_nlp = spacy.load('en_core_web_sm')

        if not self.lazy_loading:
            self.load_model()

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

    def batch_scanner_extractor(self, inputs):
        """
        Process multiple inputs (questions and/or responses) in a batch and return extracted claims.

        Args:
            inputs (list of dict): Each dict contains either:
                - {"prompt": <prompt>, "response": <response>} for non-QA inputs
                - {"prompt": <prompt>, "response": <response>} for QA inputs
            is_qa (bool): Whether the inputs are QA-based or not.
            cost_estimate_only (bool): If True, only estimate the cost without extracting claims.

        Returns:
            list of dict: Each dict contains the extracted claims for the corresponding input.
        """
        snippet_list = []
        for input_data in inputs:
            question = input_data["prompt"].strip()
            response = input_data["response"].strip()
            sentences = self.get_sentence(response)
            for i, sentence in enumerate(sentences):
                # if sentence contains less than five words, skip it
                # if len(sentence.split()) < 5:
                #     continue
                # snippet = f"Questions:\n{question}\nResponse:\n{response.replace(sentence, f'<SOS>{sentence}<EOS>')}"

                context1 = " ".join(sentences[max(0, i - 3):i])
                sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                context2 = " ".join(sentences[i + 1:i + 2])
                # remove new liens in the context
                snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
                # remove new liens in the snippet and remove all "##" and "###"
                snippet = snippet.replace("\n", " ")
                snippet = snippet.replace("####", "").replace("###", "").replace("##", "")

                snippet = f"Question: {question.strip()}\nResponse: {snippet}".strip()

                snippet_list.append({"snippet": snippet, "sentence": sentence})

        # Use batch_fact_extractor to process all snippet_list
        batch_results = self.batch_fact_extractor(snippet_list)

        # Group results back into the original input structure
        grouped_results = []
        snippet_idx = 0
        for input_data in inputs:
            response = input_data["response"].strip()

            sentences = self.get_sentence(response)
            claims_per_input = []
            for _ in sentences:
                claims_per_input.append(batch_results[snippet_idx]["claims"])
                snippet_idx += 1

            grouped_results.append({"claims": claims_per_input})

        return grouped_results

    def get_sentence(self, text):
        # use spaCy to split the text into sentences
        return [x.text.strip() for x in self.spacy_nlp(text).sents]


    def batch_fact_extractor(self, snippet_list):
        """
        Process multiple snippets in a batch and return a list of dictionaries with extracted claims.

        Args:
            snippet_list (list of dict): List of text snippets to process.

        Returns:
            list of dict: Each dict contains the extracted claims for a snippet.
        """

        if self.lazy_loading:
            self.load_model()

        if self.llm:
            from vllm import SamplingParams
            prompts = []
            results = []

            for snippet in snippet_list:
                # formatted_input = self.alpaca_prompt.format(snippet, "")
                formatted_input = get_veriscore_extractor_prompt_ft(snippet["snippet"])
                # prompt = self.get_prompt(snippet)
                formatted_input = self.tokenizer.apply_chat_template([{"role": "user", "content": formatted_input}], tokenize=False, add_generation_prompt=True)
                if self.tokenizer.bos_token and formatted_input.startswith(self.tokenizer.bos_token):
                    formatted_input = formatted_input.removeprefix(self.tokenizer.bos_token)
                
                prompts.append(formatted_input)

            outputs = self.llm.generate(
                prompts, sampling_params=SamplingParams(max_tokens=512, temperature=0, stop="\n\n")
            )

            for i, output in enumerate(outputs):
                clean_output = output.outputs[0].text
                if not clean_output or "No verifiable claim." in clean_output:
                    results.append({"claims": []})
                else:
                    claims = [x.strip() for x in clean_output.split("\n") if len(x.split()) > 1]    # at least two words
                    results.append({"claims": claims})

                # print(f"---\nSnippet: {prompts[i]}\n---")
                # print(f"---\nResponse: {clean_output}\n---")
        else:
            # results = []
            # outputs = []
            # completion_tokens = 0
            # prompt_tokens = 0
            # MAX_TRIES = 3
            # progress_bar = tqdm(total=len(snippet_list), desc="Processing Prompts", unit="prompt")
            # for snippet in snippet_list:
            #     for tries in range(MAX_TRIES):
            #         try:
            #             if self.client is None:
            #                 raise Exception("Client not initialized")
            #             response = self.client.chat.completions.create(
            #                 model=self.model_name.split("::")[-1],
            #                 messages=[
            #                     {"role": "user", "content": get_veriscore_extractor_prompt_gpt(snippet["snippet"], snippet["sentence"])}
            #                 ],
            #                 max_tokens=512,
            #                 temperature=0
            #             )
            #             outputs.append(response)
            #             if response.usage is not None:
            #                 completion_tokens += response.usage.completion_tokens
            #                 prompt_tokens += response.usage.prompt_tokens
            #             progress_bar.update(1)
            #             progress_bar.set_postfix({
            #                 "Completion Tokens": completion_tokens,
            #                 "Prompt Tokens": prompt_tokens,
            #             })
            #             break
            #         except Exception as e:
            #             print(f"Error: {e}. Retrying {tries + 1}/{MAX_TRIES}...")
            #     else:
            #         print(f"Failed to generate response for prompt: {snippet['sentence']} after {MAX_TRIES} tries.")
            #         outputs.append(None)

            # for i, output in enumerate(outputs):
            #     if output is None:
            #         results.append({"claims": []})
            #         continue
            #     response = output.choices[0].message.content
            #     clean_output = response.strip()
            #     if not clean_output or "No verifiable claim." in clean_output:
            #         results.append({"claims": []})
            #     else:
            #         claims = [x.strip() for x in clean_output.split("\n") if len(x.split()) > 1]    # at least two words
            #         results.append({"claims": claims})

            messages_list = [[{"role": "user", "content": get_veriscore_extractor_prompt_gpt(snippet["snippet"], snippet["sentence"])}] for snippet in snippet_list]
            outputs = batch_chat_complete(
                self.client,
                messages_list,
                model=self.model_name.split("::")[-1],
                max_tokens=512,
                temperature=0
            )
            results = []
            for output in outputs:
                try:
                    clean_output = output.choices[0].message.content
                    if not clean_output or "No verifiable claim." in clean_output:
                        results.append({"claims": []})
                    else:
                        claims = [x.strip() for x in clean_output.split("\n") if len(x.split()) > 1]    # at least two words
                        results.append({"claims": claims})
                    self.prompt_tokens += output.usage.prompt_tokens
                    self.completion_tokens += output.usage.completion_tokens
                except Exception as e:
                    print(f"Error: {e}")
                    results.append({"claims": []})

        if self.lazy_loading:
            self.unload_model()
            
        return results
