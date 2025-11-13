def get_customized_verifier_prompt(claim_text, passages_text):
    prompt = f"""
You need to judge whether a claim is supported or contradicted by Google search results, or whether there is no enough information to make the judgement. When doing the task, take into consideration whether the link of the search result is of a trustworthy source.

Below are the definitions of the three categories:

Supported: A claim is supported by the search results if everything in the claim is supported and nothing is contradicted by the search results. There can be some search results that are not fully related to the claim.
Contradicted: A claim is contradicted by the search results if something in the claim is contradicted by some search results. There should be no search result that supports the same part.
Inconclusive: A claim is inconclusive based on the search results if:
- a part of a claim cannot be verified by the search results,
- a part of a claim is supported and contradicted by different pieces of evidence,
- the entity/person mentioned in the claim has no clear referent (e.g., "the approach", "Emily", "a book").

>>> Begin of search results <<<
{passages_text}
<<< End of search results <<<

Claim: {claim_text}
Task: Given the search results above, is the claim supported, contradicted, or inconclusive? Your answer should be either "supported", "contradicted", or "inconclusive" without explanation and comments.

Your decision:
""".strip()
    return prompt
