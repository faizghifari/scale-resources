"""Centralized prompt/system-prompt templates for the synthetic-data generation pipeline.

Each constant is grouped by the pipeline stage that consumes it (see
synthetic_data/GENERATION_INFO.json for the full stage list). Previously these
lived inline in each script, plus one (TRANSLATION_INSTRUCTION) as a standalone
sample file (synthetic_data/instruction.txt) that only ever had its
<instruction> block read programmatically.
"""

# Stage 1: topic/subtopic/question generation (generate_topics.py)
TOPIC_SUBTOPIC_SYSTEM_PROMPT = (
    "You are an expert curriculum designer expanding high-level topics into "
    "concise, distinct subtopics. Always answer with strict JSON "
    "and the subtopics must be in Indonesian language."
)

TOPIC_QUESTION_SYSTEM_PROMPT = (
    "You are an investigative interviewer crafting thought-provoking "
    "questions in Indonesian and English contexts. Respond strictly with JSON "
    "and the questions must be in Indonesian language."
)

# Stage 2: answer generation (generate_synthetic_answers.py)
ANSWER_GENERATION_SYSTEM_PROMPT = (
    "You are an assistant who always responds in fluent, natural Indonesian. "
    "Deliver only the final answer; never describe internal reasoning or thought processes. "
    "Reduce the use of bullet points, numbered lists, headings, or any list-like formatting; only use it when really necessary. "
    "Vary sentence length and tone to keep the prose engaging, weaving in relevant examples, comparisons, or brief illustrative anecdotes when helpful. "
    "Do not repeat or rephrase the question; begin directly with the answer in Indonesian language."
)

# Stage 3: translation (translate_answers.py / run_translate_chunks.py)
TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise translation assistant. Follow every user instruction carefully. "
    "Never describe internal reasoning or thought processes. "
    "Always reply only with a single valid JSON object containing exactly two string keys: "
    '"balinese" and "cirebonese". Never add commentary, code fences, or extra text.'
)

TRANSLATION_INSTRUCTION = (
    "Given the text above in Indonesian, translate it into Balinese and Cirebonese. "
    "Translate only the content inside <text>...</text>; do not translate or echo "
    "<instruction> or any lexicon blocks. You will be given a small lexicon of "
    "candidate equivalents from Indonesian to Balinese and Cirebonese for some words "
    "from the text. Treat it only as a reference. Do not translate word-by-word. "
    "Produce a proper, high-quality translation that is grammatically correct and "
    "natural while keeping the meaning the same. You may change word order, add or "
    "omit function words, and apply appropriate affixes, particles, and pronouns to "
    "fit Balinese/Cirebonese grammar. Lexicon entries are lemmas; inflect or derive "
    "as needed. If a listed equivalent is unnatural or incorrect in context, ignore "
    "it and choose a better alternative. Use a neutral, natural register. Priority: "
    "(1) natural, idiomatic target-language grammar, (2) faithful meaning, (3) use "
    "of the lexicon where helpful. Return only a single valid, minified JSON object "
    'with exactly two string fields: "balinese" and "cirebonese"; no explanations, '
    "tags, lists, headings, or code fences; no other text before or after. Do not "
    "include analysis, chain-of-thought, justifications, alternatives, or notes. If "
    "you need to think, do so silently. If uncertain, choose the most natural single "
    "rendering and commit without hedging. Do not echo the source text or lexicon. "
    "No newlines or spaces outside of strings. Escape any double quotes inside "
    "values. Silently self-check and revise if needed; then output only the JSON."
)
