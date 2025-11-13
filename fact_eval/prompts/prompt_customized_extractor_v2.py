def get_customized_extractor_prompt(prompt_text, response_text):
    prompt = f"""
Extract fine-grained, atomic, and verifiable factual claims from the RESPONSE only. 
A valid claim is a concrete, source-checkable statement (e.g., could be checked in a database, official docs, reputable news, Wikipedia, or peer-reviewed literature).

Strict rules:

1) When to output an empty list
- If the RESPONSE expresses uncertainty, lack of knowledge, or asks for clarification (e.g., “I do not know,” “I could not find,” “not widely known,” “as of my knowledge cutoff,” “may/might/could/possibly,” “it appears,” “it is likely,” or similar hedging).
- If the RESPONSE is conditional, hypothetical, or template-like without committing to facts about a specific entity.
- If the RESPONSE lists options or guesses (“X may be a private person,” “this could be a misspelling”) rather than asserting concrete facts.
- If the RESPONSE mainly contains requests for more info, disclaimers, or meta commentary about limitations.

2) What to extract as claims
- Only declarative, unhedged facts that stand on their own and are checkable.
- Existence, properties, functions, roles, affiliations, dates, places, quantities, and relations that are stated as facts.
- If a quotation appears and is attributed with a source in the RESPONSE, you may extract the quote verbatim as a claim only if the attribution itself is factual and unhedged.

3) What NOT to extract
- Suggestions, opinions, instructions, and personal experiences.
- Speculation, possibilities, guesses, and hedged statements (any sentence using “may,” “might,” “could,” “possibly,” “appears,” “likely,” etc.).
- Negative broad claims of ignorance or search results (e.g., “there is no widely known public figure named X,” “I cannot find info on X”). Treat these as uncertainty, not facts.
- Claims from code blocks or pure math derivations.
- Summaries of the PROMPT. Extract only from the RESPONSE.

4) Atomicity and wording
- Split combined facts joined by “and” or lists into separate claims.
- If a claim can be split into smaller independent statements, split it.
- Replace pronouns with the explicit entity name if it is given in the RESPONSE; otherwise leave the pronoun unchanged.
- Write each claim exactly as stated in the RESPONSE (do not add or soften content).
- Claims must be standalone and unambiguous.

Output format:
- A JSON list of strings. Each string is exactly one atomic, verifiable claim.
- If no valid claims are present under the rules above, output [].

>>> Begin of prompt <<<
{prompt_text}
<<< End of prompt <<<

>>> Begin of response <<<
{response_text}
<<< End of response <<<

Facts (as a JSON list of strings):
""".strip()
    return prompt
