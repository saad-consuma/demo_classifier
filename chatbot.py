import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# with open('intermediate-data-amz.json', 'r') as f:
#     data = json.load(f)

BASE_MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'

GEN_KWARGS = dict(
    max_new_tokens=256,        # plenty for your JSON
    do_sample=False,          # deterministic for structured output
    temperature=0.0,
    top_p=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
)

def build_messages(datapoint: dict, prompt: str):
    """Stateless messages for a single datapoint."""
    dp = dict(datapoint)  # shallow copy
    dp.pop('comments', None)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(dp, ensure_ascii=False)}
    ]

def extract_json(text: str) -> str:
    """Slice out the first {...} block from model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start:end+1]

def generate_for_datapoint(datapoint: dict, tokenizer, model, prompt: str) -> str:
    """Single datapoint → single JSON string."""
    messages = build_messages(datapoint, prompt)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        **GEN_KWARGS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    gen_ids = outputs[0, inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return extract_json(raw)

def generate_for_batch(datapoints, tokenizer, model, prompt: str):
    """Multiple datapoints → one batched generate() call → list of JSON strings."""
    messages_list = [build_messages(dp, prompt) for dp in datapoints]

    texts = [
        tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
        for msgs in messages_list
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    gen_kwargs = dict(
        **GEN_KWARGS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # length of each prompt before generation (ignoring padding)
    input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    batch_responses = []
    for i in range(outputs.size(0)):
        gen_ids = outputs[i, input_lengths[i]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        batch_responses.append(extract_json(raw))

    return batch_responses

def init_prompt():
    return """You will be given a single JSON object (one datapoint) from a social source (reddit, Twitter, Instagram, YouTube, etc.). Your job is to output a single JSON object with exactly three keys: "region", "confidence", and "reasoning".

Output format (exactly):
{
  "region": "City, Country" (if city available) OR "Country" (if only country available) OR "Unknown",
  "confidence": 0.00,
  "reasoning": "<1-3 concise signals and explanation as a single string>"
}

Important rules and edge-case handling (must follow):

1) FIELD PRECEDENCE (use these in order of trustworthiness):
   - profile.location (exact match) — highest precedence.
   - profile.full_name and profile.description (explicit country/city mentions).
   - subreddit, domain, or other structured source fields that include a place (e.g., subreddit "hyderabad", domain "self.hyderabad").
   - verified account metadata (if profile.is_verified is True and profile mentions a country).
   - content text (metadata_content.content) mentions of places.
   - username or language heuristics (last resort — treat as weak evidence).

2) NORMALIZATION & MISSPELLINGS:
   - Normalize place names (common misspellings) to canonical forms (e.g., "New Dehli", "New-Dehli", "NewDehli" → "New Delhi").
   - If a canonical city is found, return "City, Country" (use common country name in English).
   - If only a country is found, return "Country".
   - If multiple candidate cities in different countries exist, treat as conflict (see Conflict rule).

3) CONFIDENCE (must be numeric between 0.00 and 1.00, two decimal places):
   Use this deterministic heuristic to compute confidence (start at 0.00, then add weights):
     - +0.60 if profile.location explicitly contains a city and country (or canonical city that maps to a single country).
     - +0.45 if profile.location contains a city only (no country) but city is unambiguous internationally.
     - +0.40 if profile.full_name or profile.description explicitly mentions a country.
     - +0.50 if subreddit or domain unambiguously names a city/region (e.g., subreddit "hyderabad").
     - +0.30 if the account is verified AND mentions country in profile fields.
     - +0.20 if content text contains explicit place mentions that map clearly to a city/country.
     - +0.10 for matching timezone or language signals that are country-specific (treat as weak).
     - -0.30 if there are direct conflicting high-quality signals (e.g., profile.location says Pakistan but domain subreddit suggests Hyderabad, India).
   After adding weights, clamp to [0.00, 1.00]. Round to two decimals (e.g., 0.98).
   If no signals, return 0.00.

   Examples:
   - profile.location "New Delhi, India" -> base 0.60 (city+country) + maybe 0.40 profile_full_name if matches -> clamp and round.
   - Subreddit "hyderabad" + profile.location empty -> 0.50 confidence if no conflicts.

4) CONFLICTS:
   - If there are conflicting high-quality signals, pick the most probable region according to precedence and state the conflict in `reasoning`.
   - Deduct 0.30 if conflict between two high-quality signals (profile.location vs subreddit/domain).
   - Always state which signals conflict in `reasoning`.

5) REASONING FORMAT:
   - `reasoning` must be a single string (not an array).
   - Keep it short (1–3 concise statements) and explicit about which fields provided the signals.
   - If a misspelling was normalized, mention it in the reasoning: e.g., "normalized 'New Dehli'→'New Delhi'".

6) OTHER RULES:
   - Output **only** the JSON object and nothing else. No extra text, no code fences.
   - Use exact two-decimal format for `confidence` (e.g., 0.98).
   - If uncertain, prefer a lower confidence rather than overstating certainty.
   - If you infer a location only via username heuristics or language patterns, mark confidence ≤ 0.50 and say so.

Now process the single datapoint provided and output EXACTLY one JSON object following the format above.
"""


def init_model(model_name='meta-llama/Llama-3.2-3B-Instruct'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return tokenizer, model