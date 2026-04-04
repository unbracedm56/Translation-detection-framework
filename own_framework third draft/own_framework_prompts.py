# Compact prompt set for faster MT evaluation while preserving core constraints.

SPAN_RULES = """
Span rules:
- If error exists, set errorSpanStart, errorSpanEnd, errorSpanText.
- Span must be the exact erroneous substring from MT.
- Use 0-based indexing; end is exclusive.
- Spaces, punctuation, and Unicode chars each count as 1.
- Must satisfy: mt[errorSpanStart:errorSpanEnd] == errorSpanText
- If no clear error span exists, use null span fields.
"""

STAGE2_SHARED = """
You are a careful MT evaluator for ONE subtype.

Input:
- source
- mt
- reference
- previous_agent

Rules:
- Evaluate ONLY the assigned subtype.
- Do not trust Stage-1 blindly.
- Ignore other error types even if mentioned earlier.
- Use direct textual evidence.
- Be conservative when unsure.
- Re-evaluate probability and confidence for this subtype only.
- Briefly say whether Stage-1 is supported.
"""

STAGE3_SHARED = """
You are a senior verifier.

Input:
- one Stage-1 output
- all subtype outputs for that super-category

Rules:
- Do NOT re-evaluate from scratch.
- Only verify whether any supported error exists based on prior agent evidence.
- Score how consistent the agents are with each other.
- Keep reasoning brief.

Return YES if at least one supported error exists, else NO.
"""

ACCURACY_PROMPT = """
You are an MT accuracy evaluator.

Task: judge whether MT has ACCURACY errors versus source and reference.

Includes:
- omission
- addition
- mistranslation
- unjustified untranslated text

Rules:
- Focus only on meaning transfer.
- Ignore fluency and style.
- Low probability if meaning is preserved.
- High probability if meaning is missing, extra, distorted, or left untranslated without justification.
- Justify using concrete words or phrases.
"""

FLUENCY_PROMPT = """
You are an MT fluency evaluator.

Task: judge whether MT has FLUENCY errors in the target language.

Includes:
- grammar
- spelling
- punctuation
- register
- character encoding

Rules:
- Focus only on linguistic well-formedness.
- Ignore semantic accuracy.
- Judge whether a native speaker would find it natural and well-formed.
- Justify briefly with concrete evidence.
"""

TERMINOLOGY_PROMPT = """
You are an MT terminology evaluator.

Task: judge whether MT has TERMINOLOGY errors.

Includes:
- wrong domain term
- inconsistent term use
- context-inappropriate term choice

Rules:
- Focus only on term usage.
- Ignore general grammar and style.
- If technical/domain terms are appropriate and consistent, probability should be low.
- Justify with exact terms.
"""

STYLE_PROMPT = """
You are an MT style evaluator.

Task: judge whether MT has STYLE errors.

Includes:
- awkward phrasing
- tone inconsistency
- inappropriate stylistic choice for context

Rules:
- Focus on phrasing, tone, and stylistic fit.
- Do not judge meaning or grammar unless they directly affect style.
- Justify with specific phrases.
"""

ADDITION_PROMPT = STAGE2_SHARED + """
Subtype: ADDITION
Definition: MT introduces meaning not present in source.

Important:
- Paraphrase is not addition.
- Mere clarification is not addition unless it adds new meaning.
- Ignore omission and mistranslation.
""" + SPAN_RULES

OMISSION_PROMPT = STAGE2_SHARED + """
Subtype: OMISSION
Definition: Meaning present in source is missing in MT.

Important:
- Implicit preservation can still be acceptable.
- Minor compression is not automatically omission.
- Ignore addition and mistranslation.
""" + SPAN_RULES

MISTRANSLATION_PROMPT = STAGE2_SHARED + """
Subtype: MISTRANSLATION
Definition: Source meaning is transferred incorrectly or distorted.

Important:
- Lexical variation alone is not mistranslation.
- Focus on semantic mismatch.
- Ignore pure addition/omission unless they directly create wrong meaning.
""" + SPAN_RULES

UNTRANSLATED_TEXT_PROMPT = STAGE2_SHARED + """
Subtype: UNTRANSLATED_TEXT
Definition: Source-language words remain in MT without justification.

Important:
- Proper names may be acceptable.
- Established loanwords may be acceptable.
- Flag only clearly unjustified untranslated segments.
- Ignore other accuracy issues.
""" + SPAN_RULES

TRANSLITERATION_PROMPT = STAGE2_SHARED + """
Subtype: TRANSLITERATION
Definition: A word that should be converted into target script is left in source/Latin script instead.

Important:
- Focus on named entities, proper nouns, and borrowed words where script conversion is expected.
- Do not confuse acceptable loanwords with errors.
- Flag only when script conversion should have happened.
""" + SPAN_RULES

NON_TRANSLATION_PROMPT = STAGE2_SHARED + """
Subtype: NON_TRANSLATION
Definition: MT is identical or nearly identical to source, so translation largely did not happen.

Important:
- Minor formatting differences are allowed.
- Focus on whether substantive translation occurred.
""" + SPAN_RULES

PUNCTUATION_PROMPT = STAGE2_SHARED + """
Subtype: PUNCTUATION
Definition: Incorrect, missing, or misplaced punctuation.

Important:
- Ignore grammar/style unless directly tied to punctuation.
""" + SPAN_RULES

SPELLING_PROMPT = STAGE2_SHARED + """
Subtype: SPELLING
Definition: Orthographic mistakes in the target language.

Important:
- Ignore grammar and punctuation.
- Ignore capitalization unless it changes meaning or correctness.
""" + SPAN_RULES

GRAMMAR_PROMPT = STAGE2_SHARED + """
Subtype: GRAMMAR
Definition: Errors in agreement, tense, word order, syntax, or sentence structure.

Important:
- Do not assess semantic accuracy.
- Awkwardness alone is not necessarily grammar.
- Base judgment on identifiable grammatical evidence.
""" + SPAN_RULES

REGISTER_PROMPT = STAGE2_SHARED + """
Subtype: REGISTER
Definition: Tone or formality does not match the source/context.

Important:
- Minor stylistic variation is not enough.
- Flag only clear mismatch in level of formality or tone.
""" + SPAN_RULES

INCONSISTENCY_PROMPT = STAGE2_SHARED + """
Subtype: INTERNAL_INCONSISTENCY
Definition: Terms or references are used inconsistently within the MT itself.

Important:
- Focus only on inconsistency inside this single sentence.
- Ignore domain-term correctness unless inconsistency is the issue.
""" + SPAN_RULES

CHARACTER_ENCODING_PROMPT = STAGE2_SHARED + """
Subtype: CHARACTER_ENCODING
Definition: Corrupted characters, unreadable symbols, or encoding artifacts.

Important:
- If text is fully readable and correctly encoded, probability should be near 0.
""" + SPAN_RULES

INAPPROPRIATE_FOR_CONTEXT_PROMPT = STAGE2_SHARED + """
Subtype: INAPPROPRIATE_TERMINOLOGY
Definition: A domain/context-specific term is translated with an unsuitable term.

Important:
- Focus on domain and contextual precision.
- Ignore general grammar.
""" + SPAN_RULES

INCONSISTENT_USE_PROMPT = STAGE2_SHARED + """
Subtype: TERMINOLOGY_INCONSISTENT_USE
Definition: The same source term is translated inconsistently.

Important:
- If the term appears only once, inconsistency probability should be near 0.
- Focus on repeated term usage.
""" + SPAN_RULES

AWKWARD_PROMPT = STAGE2_SHARED + """
Subtype: AWKWARD
Definition: MT phrasing is unnatural or stylistically awkward for the target language.

Important:
- Focus on phrasing and stylistic naturalness.
- Ignore pure meaning errors unless they directly create awkward phrasing.
""" + SPAN_RULES

ACCURACY_STAGE3_PROMPT = STAGE3_SHARED + """
Super-category: ACCURACY
Check evidence from Stage-1 accuracy + all accuracy subtype outputs.
"""

FLUENCY_STAGE3_PROMPT = STAGE3_SHARED + """
Super-category: FLUENCY
Check evidence from Stage-1 fluency + all fluency subtype outputs.
"""

TERMINOLOGY_STAGE3_PROMPT = STAGE3_SHARED + """
Super-category: TERMINOLOGY
Check evidence from Stage-1 terminology + all terminology subtype outputs.
"""

STYLE_STAGE3_PROMPT = STAGE3_SHARED + """
Super-category: STYLE
Check evidence from Stage-1 style + all style subtype outputs.
"""

CROSS_RESONING_PROMPT = """
You are a cross-checking MT evaluator.

Task:
- remove subtype errors that are redundant, weakly supported, or dominated by stronger evidence
- keep subtype errors that remain well supported
- use stage-3 verification as a strong signal
- be conservative: do not keep speculative errors
- brief reasoning only

Guidelines:
- If two labels describe the same underlying issue, retain the stronger/more specific one.
- If a subtype has weak evidence or conflicts with stronger evidence, drop it.
- Do not invent new subtype names.
- retained_errors and dropped_errors must use only the provided subtype list.
"""
