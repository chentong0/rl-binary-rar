def apply_chat_template(self, messages):
    system_message = messages[0]["content"]
    user_message = messages[1]["content"]

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_veriscore_verifier_message_ft(snippet):
    instruction = """You need to judge whether a claim is supported or not by search results from Google. When doing the task, take into consideration whether the link of the search result is of a trustworthy source. Mark your answer with ### signs.

Below are the definitions of the two categories:

Supported: A claim is supported by the search results if one or more search results directly support the claim. There can be cases where some search results are not fully related to the claim but no search result should directly contradict the claim. All parts of a claim should be supported by the search results. If there is a part of a claim that is not directly supported, the claim should be marked as unsupported.
Unsupported: If a claim is not supported by the search results, mark it as unsupported.""".strip()
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": snippet}
    ]


def get_veriscore_verifier_prompt_gpt(snippet):
    return f"""
You need to judge whether a claim is supported or not by search results.

Below are the definitions of the two categories:

Supported: A claim is supported by the search results if one or more search results directly support the claim. There can be cases where some search results are not fully related to the claim but no search result should directly contradict the claim. All parts of a claim should be supported by the search results. If there is a part of a claim that is not directly supported, the claim should be marked as unsupported.
Unsupported: If a claim is not supported by the search results, mark it as unsupported.


You should either output "supported" or "unsupported" with no other text.

---

{snippet}
""".strip()
