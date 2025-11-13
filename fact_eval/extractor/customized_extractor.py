import json
import os
from fact_eval.prompts.prompt_customized_extractor import get_customized_extractor_prompt
from fact_eval.utils.async_completion import batch_chat_complete


class Document:
    """A simple document class with title and text attributes."""
    
    def __init__(self, title: str, text: str):
        self.title = title
        self.text = text
    
    def __repr__(self):
        return f"Document(title='{self.title}', text='{self.text[:50]}...')"


class ClaimExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def extract(self, prompt, response):
        raise NotImplementedError


class OpenaiClaimExtractor:

    def __init__(self, model_name, batch_size=256, max_claims_per_instance=50):
        self.model_name = model_name
        self.batch_size = batch_size

        # set a limit to the max length of returned claims
        self.max_claims_per_instance = max_claims_per_instance

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

    def get_prompt(self, prompt, response):
        """Generate the extraction prompt for a given prompt-response pair."""
        return get_customized_extractor_prompt(prompt, response)

    def extract(self, prompt_list, response_list):
        task_prompt_list = [self.get_prompt(prompt, response) for prompt, response in zip(prompt_list, response_list)]
        messages_list = [[{"role": "user", "content": prompt}] for prompt in task_prompt_list]
        print(self.model_name)
        chat_completion_kwargs = {}
        decoding_kwargs = {"temperature": 0.0}
        if self.model_name.startswith("vllm-openai::"):
            # chat_completion_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}, "guided_json": {"type": "array","items": {"type": "string"}}}
            chat_completion_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            # decoding_kwargs = {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.2}
        outputs = batch_chat_complete(
            self.client,
            messages_list, batch_size=self.batch_size,
            model=self.model_name.split("::")[-1],
            max_tokens=4096,
            **decoding_kwargs,
            **chat_completion_kwargs
        )
        results = []
        import re
        for output in outputs:
            try:
                output_text = output.choices[0].message.content
                # Use regex to extract JSON from code block, handling optional 'json' and whitespace
                code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", output_text, re.IGNORECASE)
                if code_block_match:
                    output_text = code_block_match.group(1).strip()
                claim_list = json.loads(output_text)
                # if claim_list is not a list of strings, print the output_text
                if not isinstance(claim_list, list) or not all(isinstance(item, str) for item in claim_list):
                    # print(output_text)
                    # import pdb; pdb.set_trace()
                    # log a warning and use an empty list
                    print(f"Warning: Extracted claim list is not a list of strings: {output.choices[0].message.content}")
                    claim_list = []
                if self.max_claims_per_instance is not None:
                    claim_list = claim_list[:self.max_claims_per_instance]
                # Sum up token usage from all outputs
                self.prompt_tokens += output.usage.prompt_tokens
                self.completion_tokens += output.usage.completion_tokens
                results.append(claim_list)
            except:
                print(f"Error: {output.choices[0].message.content}")
                results.append([])
            
        return results

# class FactScoreClaimExtractor:
#     pass

# class VeriScoreClaimExtractor:
#     pass

# class EfficientClaimExtractor:
#     def __init__(self, model_name, api_base=None):
#         # Modify OpenAI's API key and API base to use vLLM's API server.
#         self.model_name = model_name
#         from openai import OpenAI
#         self.client = OpenAI(
#             base_url=api_base,
#         )
#         # completion = client.completions.create(model=model_name, prompt="San Francisco is a")
#         # print("Completion result:", completion)
    
#     def extract(self, prompts, responses):
#         # load extraction_prompt.py
#         from extraction_prompt import get_extraction_prompt
#         extraction_prompt_list = []
#         for prompt, response in zip(prompts, responses):
#             extraction_prompt = get_extraction_prompt(prompt, response)
#             extraction_prompt_list.append(extraction_prompt)
#         messages_list = [[{"role": "user", "content": extraction_prompt}] for extraction_prompt in extraction_prompt_list]

#         from utils_vllm import batch_chat_complete
#         results = batch_chat_complete(
#             self.client,
#             messages_list,
#             model=self.model_name,
#             max_tokens=4096,
#             temperature=0,
#         )
#         claims_list = []
#         for i, extraction_response in enumerate(results):
#             # print(f"Extraction response {i}: {extraction_response}")
#             # if extraction_response has choices attribute and it is not empty
#             if hasattr(extraction_response, 'choices') and extraction_response.choices:
#                 extraction_response = extraction_response.choices[0].message.content
#             else:
#                 extraction_response = ""
#             # extract all claims in <claim>...</claim> from the response
#             import re
#             claims = re.findall(r'<claim>(.*?)</claim>', extraction_response, re.DOTALL)
#             # remove leading and trailing whitespace from each claim
#             claims = [claim.strip() for claim in claims if claim.strip()]
#             claims_list.append(claims)
#         return claims_list


# test case
if __name__ == "__main__":
    extractor = OpenaiClaimExtractor(model_name="gpt-4.1-mini-standard")
    # extractor = OpenaiClaimExtractor(model_name="gpt-4.1-standard")
    # prompt_list = [
    #     # Math problem
    #     "Solve this calculus problem: Find the volume of the solid obtained by rotating the region bounded by y = x^2, y = 2x, and the y-axis about the x-axis.",
        
    #     # Novel writing
    #     "Write a short story about a time traveler who meets their younger self.",
        
    #     # Entity-based factual
    #     "Compare and contrast the contributions of Marie Curie and Rosalind Franklin to scientific research.",
        
    #     # Procedural knowledge
    #     "Explain the step-by-step process of performing CPR on an adult.",
        
    #     # Open-ended
    #     "What do you think will be the biggest challenges facing humanity in the next 100 years?"
    # ]
    
    # response_list = [
    #     "To solve this, we use the washer method. The outer radius is R = √y = √(2x) and inner radius is r = √y = x. The volume is V = ∫π(R² - r²)dx from x=0 to x=2. This gives us V = π∫(4x - x²)dx = π[2x² - x³/3]₀² = 16π/3 cubic units.",
        
    #     "Sarah stood face-to-face with her 12-year-old self in 1995. 'Don't give up the violin,' she whispered, knowing it would change everything. Her younger self nodded, eyes wide with wonder. The temporal paradox detector on her wrist began to beep urgently. She had 30 seconds before the timeline would collapse.",
        
    #     "Marie Curie discovered radium and polonium, winning Nobel Prizes in both Physics and Chemistry. She pioneered research in radioactivity. Franklin's X-ray diffraction images were crucial to understanding DNA's structure, particularly Photo 51. Both faced significant gender discrimination in their careers.",
        
    #     "First, check scene safety and victim responsiveness. Call 911. Check breathing for 10 seconds. Begin compressions: 30 compressions at 100-120 beats per minute, pushing 2-2.4 inches deep. Give 2 rescue breaths. Continue 30:2 ratio until help arrives or victim responds.",
        
    #     "Humanity will likely face severe climate change impacts, including rising sea levels and extreme weather. AI and automation may disrupt employment patterns. Resource scarcity could lead to global conflicts. Pandemics may become more frequent due to population density and climate change."
    # ]
    
    prompt_list = [
        "In a paragraph, could you tell me what you know about AHKCvJoyInterface?"
    ]

    response_list = [
        "As of my last update in 2023, AHKCvJoyInterface may be a specialized tool or script typically associated with the Advanced HOTAS Integrator (AHK) used for homebuilt haptic joysticks' integration into other devices like PC peripherals or automotive interfaces. AHK isn't well-documented publicly; however, its developers or some online forums might provide more context or usage instructions. Users might employ AHKCvJoyInterface to reconfigure or simulate input or feedback for joystick controls across a range of platforms or devices that don't traditionally support joysticks' use. It may be a custom solution allowing for fine-grained control over digital gamepads, input devices, or similar technologies, enabling users to either simulate other forms of input or potentially enhance real-time feedback during tasks requiring precise joystick operation, gaming, simulations, or developing software support for non-standard peripherals. However, specific details on its implementation, functionality, or community knowledge might require reaching out to dedicated communities or forums for Advanced HOTAS Integrator or its counterparts.\n"
    ]

    results = extractor.extract(prompt_list, response_list)
    print("\nExtracted claims from diverse response types:")
    for i, (prompt, claims) in enumerate(zip(prompt_list, results)):
        print(f"\nCategory {i+1}:")
        print(f"Prompt: {prompt}")
        print("Claims:")
        for j, claim in enumerate(claims):
            print(f"  {j+1}. {claim}")