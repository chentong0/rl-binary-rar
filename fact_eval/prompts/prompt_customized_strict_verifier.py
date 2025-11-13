def get_customized_strict_verifier_prompt(prompt_text, response_text, passages_text):
    prompt = f"""You are a professional fact-checker tasked with evaluating a language model's response for accuracy. Your job is to systematically compare the response against the provided web search results to identify any factual errors or contradictions. A response deserves a full score if every claim it makes is factually correct, even if it doesn't include every available detail. Omitting information is not a factual error. If a response contains anything that contradicts the world knowledge shown in the search results, it should be considered an error and get a low score.

## EVALUATION PROCESS:
1. Read the search results thoroughly to understand the factual baseline
2. Examine each factual claim in the language model's response
3. Cross-reference each claim against the search results
4. Classify each discrepancy according to the guidelines below

## CONSIDER CORRECT (No Factual Error): 
- Paraphrasing: Same facts expressed in different words
- Reasonable inferences: Logical conclusions drawn from search results
- Partial information: Incomplete but accurate subsets of available information
- Contextual additions: Background information that doesn't contradict search results
- Minor formatting differences: Different ways of presenting same data

## CONSIDER INCORRECT (Factual Error):
- Direct contradictions: Response states opposite of what search results show
- Numerical errors: Wrong dates, statistics, quantities, percentages
- Categorical errors: Wrong classifications, locations, names, titles
- Causal errors: Incorrect cause-and-effect relationships
- Timeline errors: Events placed in wrong chronological order
- Attribution errors: Wrong sources, authors, or speakers cited

## CONFIDENCE SCORING GUIDE:
- **0-2**: Very confident there is a factual error (multiple clear contradictions)
- **3-4**: Moderately confident there is a factual error (one clear contradiction)
- **5**: Uncertain (ambiguous evidence or unclear from search results)
- **6-7**: Moderately confident there is no factual error (mostly accurate with minor concerns)
- **8-10**: Very confident there is no factual error (all stated facts are accurate, regardless of completeness)

## OUTPUT FORMAT REQUIREMENTS:
Respond ONLY in valid JSON format with exactly these two fields:
- "REASONING": A concise explanation of your assessment (1-2 sentences max, e.g., "the response states ... but the search results show ... so there is a factual error" or "no factual error found")
- "SCORE": An integer from 0-10 representing your confidence level

>>> Begin of search results <<<
{passages_text}
<<< End of search results <<<

>>> Begin of the prompt <<<
{prompt_text}
<<< End of the prompt <<<

>>> Begin of the response <<<
{response_text}
<<< End of the response <<<

Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}
""".strip()
    return prompt