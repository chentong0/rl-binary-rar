import os
from fact_eval.utils.async_completion import batch_chat_complete
from fact_eval.prompts.prompt_customized_strict_verifier import get_customized_strict_verifier_prompt
from fact_eval.prompts.prompt_customized_strict_binary_verifier import get_customized_strict_binary_verifier_prompt

class OpenaiStrictVerifier:
    def __init__(self, model_name, prompt_type="rating", batch_size=256):
        self.model_name = model_name
        self.batch_size = batch_size
        self.prompt_type = prompt_type # "rating" or "binary"
        
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

    def get_prompt(self, prompt, response, passages):
        if self.prompt_type == "rating":
            return get_customized_strict_verifier_prompt(prompt, response, passages)
        elif self.prompt_type == "binary":
            return get_customized_strict_binary_verifier_prompt(prompt, response, passages)


    def verify(self, prompt_list, response_list, passages_list):
        task_prompt_list = [
            self.get_prompt(prompt, response, "\n".join([f"Doc [{i}] {p}" for i, p in enumerate(passages)]))
            for prompt, response, passages in zip(prompt_list, response_list, passages_list)
        ]
        messages_list = [[{"role": "user", "content": prompt}] for prompt in task_prompt_list]
        chat_completion_kwargs = {}
        if self.model_name.startswith("vllm-openai::"):
            chat_completion_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        outputs = batch_chat_complete(
            self.client,
            messages_list, batch_size=self.batch_size,
            model=self.model_name.split("::")[-1],
            max_tokens=1024,
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
                # import pdb; pdb.set_trace()
                # results.append(correctness)

                # output_text = output.choices[0].message.content

                output_text = output.choices[0].message.content
                import re
                import json
                # Use regex to extract JSON from code block, handling optional 'json' and whitespace
                code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", output_text, re.IGNORECASE)
                if code_block_match:
                    output_text = code_block_match.group(1).strip()
                result_dict = json.loads(output_text)
                # if claim_list is not a list of strings, print the output_text
                if not isinstance(result_dict, dict) or "REASONING" not in result_dict or "SCORE" not in result_dict:
                    print(f"Warning: Extracted result dict is not a dict with 'REASONING' and 'SCORE': {output.choices[0].message.content}")
                    result_dict = {"REASONING": None, "SCORE": 0}
                # convert SCORE to a float number between 0 and 10, throw error if not possible
                if not isinstance(result_dict["SCORE"], (int, float)) or not (0 <= result_dict["SCORE"] <= 10):
                    print(f"Warning: SCORE is not a number: {output.choices[0].message.content}")
                    result_dict = {"REASONING": None, "SCORE": 0}
                result_dict["SCORE"] = float(result_dict["SCORE"])
                results.append(result_dict)
            except Exception as e:
                print(f"Error: {e}")
                results.append({"REASONING": None, "SCORE": 0})
        
        return results

