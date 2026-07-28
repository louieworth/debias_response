# Consolidated Simulation Script

# --- IMPORTS ---
import json
import os
import pandas as pd
import time
import openai
import re
from dotenv import load_dotenv
import sys
from tqdm import tqdm
import concurrent.futures
import threading
from functools import partial
from urllib import error, request

# --- CONFIGURATION ---
# Adjust this based on your API rate limits and machine's capability
DEFAULT_MAX_WORKERS = 4

# Number of simulations per participant (default: 1)
# Set to 5 to get 4 additional simulations per participant
NUM_SIMULATIONS_PER_PARTICIPANT = 1

# Temperature for LLM (higher = more diverse responses)
# Default: 0.2, increase for more varied responses across simulations
TEMPERATURE = 0.7

# Send the survey in smaller batches to stay within TPM limits.
QUESTIONS_PER_REQUEST = 12

# Automatic retries for transient rate limits / network failures.
MAX_API_RETRIES = 6
INITIAL_RETRY_SLEEP_SECONDS = 15
MAX_RETRY_SLEEP_SECONDS = 120

# Select the concrete model here.
# API keys should still be provided via .env.
# Examples:
# - "claude-opus-4-6"
# - "doubao-seed-2.0-pro"
# - "glm-4.7"
# - "deepseek-v3.2"
# - "qwen3-max"
# - "gpt-4o"
# - "gpt-5.4"
LLM_VARIANT = os.getenv("LLM_VARIANT", "gpt-4o")

# --- PATH SETUP ---
# Dynamically determine the project root and dataset directory.
# This makes the script runnable from any location.
try:
    # Assumes the script is in a subdirectory of the project root.
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # Fallback for interactive environments or when __file__ is not defined.
    PROJECT_ROOT = os.path.abspath('.')

DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'twin_simulation')

# Add project root to sys.path for robust imports if ever needed
sys.path.insert(0, PROJECT_ROOT)


# --- LLM LOGIC ---
load_dotenv()

# Maximum completion length for a single simulation request
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))

# Per-request timeout in seconds
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

SUPPORTED_PROVIDERS = {"claude", "ark", "qwen", "openai"}

PROVIDER_DEFAULT_MAX_WORKERS = {
    "claude": 1,
    "ark": 4,
    "qwen": 2,
    "openai": 4,
}

PROVIDER_DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = {
    "claude": 1.0,
    "ark": 0.25,
    "qwen": 0.5,
    "openai": 0.25,
}

QWEN_BASE_URLS = {
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

ARK_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

ARK_MODEL_ALIASES = {
    # Coding-Plan style names -> regular online inference model IDs.
    "doubao-seed-2.0-pro": "doubao-seed-2-0-pro-260215",
    "glm-4.7": "glm-4-7-251222",
    "deepseek-v3.2": "deepseek-v3-2-251201",
}


def _parse_bool(value, default=False):
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _slugify(text):
    return re.sub(r"[^a-z0-9._-]+", "_", text.strip().lower()).strip("_")


def _resolve_ark_model_name(variant_key):
    return ARK_MODEL_ALIASES.get(variant_key, variant_key)


LLM_VARIANT_KEY = LLM_VARIANT.strip().lower()
LLM_PROVIDER_OVERRIDE = os.getenv("LLM_PROVIDER", "").strip().lower() or None

if LLM_PROVIDER_OVERRIDE:
    if LLM_PROVIDER_OVERRIDE not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER_OVERRIDE!r}. "
            f"Choose from: {sorted(SUPPORTED_PROVIDERS)}"
        )
    LLM_PROVIDER = LLM_PROVIDER_OVERRIDE
    MODEL_NAME = _resolve_ark_model_name(LLM_VARIANT_KEY) if LLM_PROVIDER == "ark" else LLM_VARIANT.strip()
elif LLM_VARIANT_KEY.startswith("claude"):
    LLM_PROVIDER = "claude"
    MODEL_NAME = LLM_VARIANT_KEY
elif (
    LLM_VARIANT_KEY.startswith("doubao")
    or LLM_VARIANT_KEY.startswith("glm-")
    or LLM_VARIANT_KEY.startswith("deepseek-")
):
    LLM_PROVIDER = "ark"
    MODEL_NAME = _resolve_ark_model_name(LLM_VARIANT_KEY)
elif LLM_VARIANT_KEY.startswith("qwen"):
    LLM_PROVIDER = "qwen"
    MODEL_NAME = LLM_VARIANT_KEY
elif LLM_VARIANT_KEY.startswith("gpt-"):
    LLM_PROVIDER = "openai"
    MODEL_NAME = LLM_VARIANT_KEY
else:
    raise ValueError(
        f"Unsupported LLM_VARIANT={LLM_VARIANT!r}. "
        "Use a concrete model name such as 'claude-opus-4-6', "
        "'doubao-seed-2.0-pro', 'glm-4.7', 'deepseek-v3.2', 'qwen3-max', "
        "'gpt-4o', or 'gpt-5'."
    )

if LLM_PROVIDER not in SUPPORTED_PROVIDERS:
    raise ValueError(
        f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. "
        f"Choose from: {sorted(SUPPORTED_PROVIDERS)}"
    )

LLM_VARIANT_SLUG = _slugify(LLM_VARIANT) or _slugify(f"{LLM_PROVIDER}_{MODEL_NAME}")

MAX_WORKERS = int(
    os.getenv(
        "LLM_MAX_WORKERS",
        str(PROVIDER_DEFAULT_MAX_WORKERS.get(LLM_PROVIDER, DEFAULT_MAX_WORKERS)),
    )
)
MIN_REQUEST_INTERVAL_SECONDS = float(
    os.getenv(
        "LLM_MIN_REQUEST_INTERVAL_SECONDS",
        str(PROVIDER_DEFAULT_MIN_REQUEST_INTERVAL_SECONDS.get(LLM_PROVIDER, 0.0)),
    )
)

QWEN_REGION = os.getenv("QWEN_REGION", "beijing").strip().lower()
if LLM_PROVIDER == "qwen" and QWEN_REGION not in QWEN_BASE_URLS:
    raise ValueError(f"Unsupported QWEN_REGION={QWEN_REGION!r}. Choose from: {sorted(QWEN_BASE_URLS)}")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL") or QWEN_BASE_URLS.get(QWEN_REGION, QWEN_BASE_URLS["beijing"])
QWEN_ENABLE_THINKING = _parse_bool(os.getenv("QWEN_ENABLE_THINKING", "false"), default=False)
if LLM_PROVIDER == "qwen" and QWEN_ENABLE_THINKING:
    raise ValueError(
        "QWEN_ENABLE_THINKING=true is incompatible with response_format=json_object "
        "in this script. Set QWEN_ENABLE_THINKING=false."
    )

ARK_BASE_URL = os.getenv("ARK_BASE_URL") or os.getenv("DOUBAO_BASE_URL") or ARK_DEFAULT_BASE_URL
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower() or None

JSON_OUTPUT_INSTRUCTION = (
    "\n\nReturn only a single valid JSON object.\n"
    "Requirements:\n"
    "- Use question keys exactly as provided, for example Q1, Q2, ...\n"
    "- The top-level JSON object itself must be the answer map; do not wrap it inside answers/result/data\n"
    "- Do not include markdown, code fences, commentary, or extra text\n"
    "- Every value must be the answer for that question\n"
)

DEBUG_RESPONSE_LOG = os.path.join(OUTPUT_DIR, f"debug_responses_{LLM_VARIANT_SLUG}.jsonl")
_REQUEST_THROTTLE_LOCK = threading.Lock()
_LAST_REQUEST_STARTED_AT = 0.0


def _resolve_api_key(provider):
    if provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
        env_name = "ANTHROPIC_API_KEY"
    elif provider == "ark":
        api_key = os.getenv("ARK_API_KEY") or os.getenv("LLM_API_KEY")
        env_name = "ARK_API_KEY"
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        env_name = "OPENAI_API_KEY"
    else:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
        env_name = "DASHSCOPE_API_KEY"

    if not api_key:
        raise ValueError(
            f"Missing API key for provider={provider}. "
            f"Set LLM_API_KEY or {env_name} in your .env file."
        )
    return api_key


def _build_json_prompt(prompt):
    return prompt + JSON_OUTPUT_INSTRUCTION


def _serialize_for_debug(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_for_debug(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_debug(v) for v in value]
    if hasattr(value, "model_dump"):
        return _serialize_for_debug(value.model_dump())
    if hasattr(value, "to_dict"):
        return _serialize_for_debug(value.to_dict())
    return repr(value)


def _write_debug_event(stage, debug_context=None, prompt=None, raw_response=None, error_message=None):
    if LLM_PROVIDER != "qwen":
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": LLM_PROVIDER,
        "model": MODEL_NAME,
        "stage": stage,
        "context": _serialize_for_debug(debug_context),
        "error_message": error_message,
        "prompt_chars": len(prompt) if prompt is not None else None,
        "prompt_preview": prompt[:1000] if isinstance(prompt, str) else None,
        "raw_response": _serialize_for_debug(raw_response),
    }
    with open(DEBUG_RESPONSE_LOG, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _throttle_request_start():
    global _LAST_REQUEST_STARTED_AT
    if MIN_REQUEST_INTERVAL_SECONDS <= 0:
        return

    with _REQUEST_THROTTLE_LOCK:
        now = time.monotonic()
        wait_seconds = (_LAST_REQUEST_STARTED_AT + MIN_REQUEST_INTERVAL_SECONDS) - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.monotonic()
        _LAST_REQUEST_STARTED_AT = now


def _flatten_survey_questions(survey):
    if isinstance(survey, list):
        return survey

    questions = []
    for category, items in survey.items():
        if category == "Demographics":
            continue
        questions.extend(items)
    return questions


def _chunk_list(items, chunk_size):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _get_openai_reasoning_effort(model):
    if OPENAI_REASONING_EFFORT:
        return OPENAI_REASONING_EFFORT
    if model.startswith("gpt-5.4"):
        return "none"
    if model.startswith("gpt-5.1"):
        return "none"
    if model.startswith("gpt-5-pro"):
        return "high"
    if model.startswith("gpt-5"):
        return "minimal"
    return None


def _looks_like_misaligned_output(df):
    required_columns = {"simulation_id", "ParticipantID", "Race", "Gender", "Age"}
    if df.empty or not required_columns.issubset(df.columns):
        return False

    sample = df.iloc[0]
    simulation_id = str(sample.get("simulation_id", ""))
    participant_id = str(sample.get("ParticipantID", ""))

    return ("-" in simulation_id) and participant_id.isdigit()


def _extract_json_object(text):
    if text is None:
        return None

    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _looks_like_question_key(key):
    return isinstance(key, str) and re.fullmatch(r"Q\d+", key) is not None


def _normalize_list_answer_map(items):
    answer_map = {}
    for item in items:
        if not isinstance(item, dict):
            return None

        question_key = (
            item.get("question_id")
            or item.get("id")
            or item.get("key")
            or item.get("question")
        )
        answer_value = (
            item.get("answer")
            if "answer" in item
            else item.get("value")
            if "value" in item
            else item.get("response")
        )

        if not _looks_like_question_key(question_key):
            return None
        answer_map[question_key] = answer_value

    return answer_map or None


def _normalize_question_answer_map(response_obj):
    if isinstance(response_obj, list):
        return _normalize_list_answer_map(response_obj)

    if not isinstance(response_obj, dict):
        return None

    if response_obj and all(_looks_like_question_key(key) for key in response_obj.keys()):
        return response_obj

    if len(response_obj) == 1:
        inner_value = next(iter(response_obj.values()))
        if isinstance(inner_value, str):
            try:
                inner_value = json.loads(inner_value)
            except json.JSONDecodeError:
                inner_value = None
        normalized_inner = _normalize_question_answer_map(inner_value)
        if normalized_inner is not None:
            return normalized_inner

    for value in response_obj.values():
        normalized_inner = _normalize_question_answer_map(value)
        if normalized_inner is not None:
            return normalized_inner

    return None


def _http_post_json(url, headers, payload):
    _throttle_request_start()
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc


def _get_qwen_client():
    return openai.OpenAI(
        api_key=_resolve_api_key("qwen"),
        base_url=QWEN_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
    )


def _get_openai_client():
    client_kwargs = {
        "api_key": _resolve_api_key("openai"),
        "timeout": LLM_TIMEOUT_SECONDS,
    }
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    return openai.OpenAI(**client_kwargs)


def _call_claude(prompt, model, temperature):
    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": temperature,
        "messages": [{"role": "user", "content": _build_json_prompt(prompt)}],
    }
    headers = {
        "x-api-key": _resolve_api_key("claude"),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response = _http_post_json(ANTHROPIC_MESSAGES_URL, headers, payload)
    content = response.get("content", [])
    text_chunks = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "\n".join(text_chunks).strip()


def _call_ark(prompt, model, temperature):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _build_json_prompt(prompt)}],
        "temperature": temperature,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {_resolve_api_key('ark')}",
        "Content-Type": "application/json",
    }
    url = f"{ARK_BASE_URL.rstrip('/')}/chat/completions"
    response = _http_post_json(url, headers, payload)
    return response["choices"][0]["message"]["content"].strip()


def _call_qwen(prompt, model, temperature):
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": _build_json_prompt(prompt)}],
        "temperature": temperature,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }

    if model.startswith("qwen3"):
        request_kwargs["extra_body"] = {"enable_thinking": QWEN_ENABLE_THINKING}

    _throttle_request_start()
    response = _get_qwen_client().chat.completions.create(**request_kwargs)
    message = response.choices[0].message
    debug_payload = {
        "message": _serialize_for_debug(message),
        "finish_reason": _serialize_for_debug(getattr(response.choices[0], "finish_reason", None)),
        "usage": _serialize_for_debug(getattr(response, "usage", None)),
    }
    return (message.content or "").strip(), debug_payload


def _call_openai_responses(prompt, model, temperature):
    request_kwargs = {
        "model": model,
        "input": [{"role": "user", "content": _build_json_prompt(prompt)}],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "text": {"format": {"type": "json_object"}},
    }

    reasoning_effort = _get_openai_reasoning_effort(model)
    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    _throttle_request_start()
    response = _get_openai_client().responses.create(**request_kwargs)
    status = getattr(response, "status", None)
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) if details is not None else None
        partial = (getattr(response, "output_text", None) or "").strip()
        if partial:
            return partial
        raise RuntimeError(f"OpenAI Responses API incomplete response: {reason or 'unknown reason'}")
    return (getattr(response, "output_text", None) or "").strip()


def _call_openai_chat_completions(prompt, model, temperature):
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": _build_json_prompt(prompt)}],
        "temperature": temperature,
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }

    reasoning_effort = _get_openai_reasoning_effort(model)
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort

    _throttle_request_start()
    response = _get_openai_client().chat.completions.create(**request_kwargs)
    message = response.choices[0].message
    return (message.content or "").strip()


def _call_openai(prompt, model, temperature):
    if model.startswith("gpt-5"):
        return _call_openai_responses(prompt, model, temperature)
    return _call_openai_chat_completions(prompt, model, temperature)


def get_json_response(prompt, model=MODEL_NAME, temperature=TEMPERATURE, debug_context=None):
    """
    Query the configured LLM provider and parse the returned JSON object.
    """
    sleep_seconds = INITIAL_RETRY_SLEEP_SECONDS

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            raw_debug_payload = None
            if LLM_PROVIDER == "claude":
                response_content = _call_claude(prompt, model=model, temperature=temperature)
            elif LLM_PROVIDER == "ark":
                response_content = _call_ark(prompt, model=model, temperature=temperature)
            elif LLM_PROVIDER == "openai":
                response_content = _call_openai(prompt, model=model, temperature=temperature)
            else:
                response_content, raw_debug_payload = _call_qwen(prompt, model=model, temperature=temperature)

            parsed = _extract_json_object(response_content)
            if parsed is None:
                _write_debug_event(
                    stage="non_json_response",
                    debug_context=debug_context,
                    prompt=prompt,
                    raw_response={
                        "response_content": response_content,
                        "provider_payload": raw_debug_payload,
                    },
                    error_message="Model returned non-JSON content",
                )
                raise ValueError(f"Model returned non-JSON content: {response_content[:500]}")
            return parsed
        except Exception as e:
            error_message = str(e)
            is_retryable = (
                "HTTP 429" in error_message
                or "rate_limit" in error_message.lower()
                or "overloaded" in error_message.lower()
                or "timeout" in error_message.lower()
                or "network error" in error_message.lower()
            )

            if attempt >= MAX_API_RETRIES or not is_retryable:
                _write_debug_event(
                    stage="request_failed",
                    debug_context=debug_context,
                    prompt=prompt,
                    raw_response={"last_error": error_message},
                    error_message=error_message,
                )
                print(f"Error getting JSON response from LLM provider={LLM_PROVIDER}, model={model}: {e}")
                return None

            print(
                f"Retryable API error from provider={LLM_PROVIDER}, model={model} "
                f"(attempt {attempt}/{MAX_API_RETRIES}): {e}. "
                f"Sleeping {sleep_seconds}s before retry."
            )
            time.sleep(sleep_seconds)
            sleep_seconds = min(sleep_seconds * 2, MAX_RETRY_SLEEP_SECONDS)

    return None

# --- SIMULATION LOGIC ---

def load_participant_pool(filename):
    """Loads the participant pool from a CSV file."""
    return pd.read_csv(filename)

def simulate_simple_response(participant, survey, survey_prompt_template, temperature=TEMPERATURE):
    """
    Simulates a survey response for a given participant using a simple persona.
    """
    participant_info = {
        "Age": participant.get("Age", "unknown"),
        "Gender": participant.get("Gender", "unknown"),
        "Race": participant.get("Race", "unknown")
    }

    prompt_with_persona = survey_prompt_template.replace("$age", str(participant_info["Age"])) \
                                                .replace("$gender", participant_info["Gender"]) \
                                                .replace("$race", participant_info["Race"])

    survey_questions = _flatten_survey_questions(survey)
    final_response = {
        question_data["Variable_Name"]: None
        for question_data in survey_questions
    }

    for question_chunk in _chunk_list(survey_questions, QUESTIONS_PER_REQUEST):
        questions_str = ""
        question_map = {}
        for idx, question_data in enumerate(question_chunk, start=1):
            q_key = f"Q{idx}"
            original_id = question_data['Variable_Name']
            question_text = question_data['Question']

            questions_str += f"{q_key}: {question_text}\n"
            question_map[q_key] = original_id

        full_prompt = prompt_with_persona + "\n" + questions_str
        llm_response_json = get_json_response(full_prompt, temperature=temperature)

        if llm_response_json is None:
            return None

        normalized_answers = _normalize_question_answer_map(llm_response_json)
        if normalized_answers is None:
            print(
                f"Error normalizing answer map from provider={LLM_PROVIDER}, model={MODEL_NAME}. "
                f"Top-level keys: {list(llm_response_json)[:5] if isinstance(llm_response_json, dict) else type(llm_response_json).__name__}"
            )
            return None

        for q_key, answer in normalized_answers.items():
            if q_key in question_map:
                original_id = question_map[q_key]
                final_response[original_id] = answer

    return final_response


def process_participant(participant, survey, survey_prompt_template, failure_log_path, simulation_ids=None):
    """
    Worker function to process a single participant with multiple simulations.
    Handles retries and returns a list of response dictionaries (one per simulation).
    """
    participant_id = participant['ParticipantID']
    all_simulations_results = []
    if simulation_ids is None:
        simulation_ids = list(range(1, NUM_SIMULATIONS_PER_PARTICIPANT + 1))

    for simulation_id in simulation_ids:
        retries = 3
        for attempt in range(retries):
            try:
                # A small delay to avoid hitting rate limits too hard
                time.sleep(attempt * 0.5)

                # Use slightly different temperature for each simulation to get diverse responses
                # Base temperature + small variation based on simulation ID
                sim_temperature = TEMPERATURE + ((simulation_id - 1) * 0.1)

                survey_response = simulate_simple_response(
                    participant,
                    survey,
                    survey_prompt_template,
                    temperature=sim_temperature
                )

                if survey_response is None:
                    raise Exception("LLM call failed and returned None.")

                full_response = participant.to_dict()
                full_response['simulation_id'] = simulation_id
                full_response['llm_variant'] = LLM_VARIANT
                full_response['llm_provider'] = LLM_PROVIDER
                full_response['llm_model'] = MODEL_NAME
                full_response.update(survey_response)
                all_simulations_results.append(full_response)
                break  # Success, move to next simulation

            except Exception as e:
                tqdm.write(
                    f"Error for participant {participant_id}, simulation {simulation_id} "
                    f"(Attempt {attempt+1}/{retries}): {e}"
                )
                if attempt >= retries - 1:
                    tqdm.write(
                        f"Failed to process participant {participant_id}, simulation {simulation_id} "
                        f"after {retries} attempts. Logging."
                    )
                    with open(failure_log_path, 'a') as f:
                        f.write(f"{participant_id}_sim{simulation_id}\n")
                    break  # Move to next simulation

    return all_simulations_results if all_simulations_results else None


def process_participant_task(task, survey, survey_prompt_template, failure_log_path):
    participant, simulation_ids = task
    return process_participant(
        participant,
        survey,
        survey_prompt_template,
        failure_log_path,
        simulation_ids=simulation_ids,
    )

# --- MAIN EXECUTION LOGIC ---

def main():
    print("--- Starting Consolidated Survey Simulation (Parallel) ---")
    print(f"Variant: {LLM_VARIANT}")
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Model: {MODEL_NAME}")
    print(f"Questions per request: {QUESTIONS_PER_REQUEST}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Min request interval seconds: {MIN_REQUEST_INTERVAL_SECONDS}")
    if LLM_PROVIDER == "qwen":
        print(f"Qwen base URL: {QWEN_BASE_URL}")
        print(f"Qwen enable thinking: {QWEN_ENABLE_THINKING}")
    if LLM_PROVIDER == "ark":
        print(f"Ark base URL: {ARK_BASE_URL}")
    if LLM_PROVIDER == "openai":
        print(f"OpenAI base URL: {OPENAI_BASE_URL or 'default'}")
        reasoning_effort = _get_openai_reasoning_effort(MODEL_NAME)
        if reasoning_effort is not None:
            print(f"OpenAI reasoning effort: {reasoning_effort}")
    
    # Path setup is now in the global scope, but file paths are defined here
    PARTICIPANT_FILE = os.path.join(DATASET_DIR, 'participant_pool.csv')
    SURVEY_FILE = os.path.join(DATASET_DIR, 'final_question_sheet_with_embedding.json')
    TEMPLATE_FILE = os.path.join(DATASET_DIR, 'survey_response_template.txt')
    output_filename = os.getenv(
        "SIMULATION_OUTPUT_FILE",
        f"simulated_responses_consolidated_{LLM_VARIANT_SLUG}.csv"
    )
    failure_log_filename = os.getenv(
        "SIMULATION_FAILURE_LOG",
        f"failed_participants_{LLM_VARIANT_SLUG}.log"
    )
    OUTPUT_FILE = os.path.join(DATASET_DIR, output_filename)
    FAILURE_LOG = os.path.join(OUTPUT_DIR, failure_log_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verify that all input files exist before starting
    for f in [PARTICIPANT_FILE, SURVEY_FILE, TEMPLATE_FILE]:
        if not os.path.exists(f):
            print(f"Error: Required input file not found at {f}")
            return

    # Load Data
    print(f"Loading participant pool from {PARTICIPANT_FILE}...")
    participant_pool = pd.read_csv(PARTICIPANT_FILE)
    
    print(f"Loading survey from {SURVEY_FILE}...")
    with open(SURVEY_FILE, 'r') as f:
        survey = json.load(f)
        
    print(f"Loading prompt template from {TEMPLATE_FILE}...")
    with open(TEMPLATE_FILE, 'r') as f:
        survey_prompt_template = f.read()

    # Prepare for Simulation
    question_headers = [q['Variable_Name'] for q in _flatten_survey_questions(survey)]

    output_columns = ['simulation_id', 'llm_variant', 'llm_provider', 'llm_model'] + list(participant_pool.columns) + question_headers

    if not os.path.exists(OUTPUT_FILE):
        print(f"Creating new output file: {OUTPUT_FILE}")
        pd.DataFrame(columns=output_columns).to_csv(OUTPUT_FILE, index=False)

    try:
        processed_df = pd.read_csv(OUTPUT_FILE)
        if _looks_like_misaligned_output(processed_df):
            print(
                f"Error: Existing output file appears misaligned and cannot be resumed safely: {OUTPUT_FILE}\n"
                "This usually happens when an older row order does not match the current CSV header.\n"
                "Please rename or remove this file, then rerun the script."
            )
            return
        # Check for both ParticipantID and simulation_id
        if 'simulation_id' in processed_df.columns:
            # Track which (participant, simulation) pairs have been processed
            processed_pairs = set(
                zip(
                    processed_df['ParticipantID'].tolist(),
                    processed_df['simulation_id'].tolist(),
                )
            )
            processed_participants = processed_df['ParticipantID'].unique().tolist()
        else:
            # Old format without simulation_id, migrate to new format
            processed_pairs = set()
            processed_participants = processed_df['ParticipantID'].tolist()
    except (pd.errors.EmptyDataError, FileNotFoundError, KeyError):
        processed_pairs = set()
        processed_participants = []

    # Only process participants that haven't been fully processed
    # (i.e., all NUM_SIMULATIONS simulations completed)
    participants_to_process = []
    total_missing_simulations = 0
    for index, row in participant_pool.iterrows():
        participant_id = row['ParticipantID']
        missing_simulation_ids = [
            simulation_id
            for simulation_id in range(1, NUM_SIMULATIONS_PER_PARTICIPANT + 1)
            if (participant_id, simulation_id) not in processed_pairs
        ]
        if missing_simulation_ids:
            participants_to_process.append((row, missing_simulation_ids))
            total_missing_simulations += len(missing_simulation_ids)

    print(f"\nFound {len(participant_pool)} total participants.")
    print(f"{len(processed_participants)} unique participants have at least one simulation.")
    print(f"Each participant will have {NUM_SIMULATIONS_PER_PARTICIPANT} simulations.")
    print(f"Remaining simulations to run: {total_missing_simulations}")

    if len(participants_to_process) == 0:
        print("All participants have been fully processed. Simulation complete.")
        return

    print(f"Starting parallel simulation for {len(participants_to_process)} participants needing more simulations.")
    print(f"Total simulations to run: {total_missing_simulations}")
    print(f"Using {MAX_WORKERS} workers.")
    print("\n*** IMPORTANT: This process may take a long time and incur API costs. ***\n")
    time.sleep(3)

    # Create a list of participant data to process
    tasks = participants_to_process

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use functools.partial to create a function with fixed arguments for the survey, template, and log path
        worker_func = partial(
            process_participant_task,
            survey=survey,
            survey_prompt_template=survey_prompt_template,
            failure_log_path=FAILURE_LOG,
        )

        # Use executor.map to run the tasks in parallel and collect results
        # The tqdm wrapper will show progress as tasks are completed
        results_iterator = executor.map(worker_func, tasks)

        for result_list in tqdm(results_iterator, total=len(tasks), desc="Simulating Responses"):
            if result_list is not None:
                # result_list is a list of simulation results for one participant
                # Filter out simulations that were already processed
                new_results = []
                for result in result_list:
                    participant_id = result['ParticipantID']
                    sim_id = result['simulation_id']
                    pair_key = (participant_id, sim_id)

                    # Only append if this (participant, simulation) pair hasn't been processed
                    if pair_key not in processed_pairs:
                        new_results.append(result)
                        processed_pairs.add(pair_key)

                # Append new successful results to the CSV
                if new_results:
                    pd.DataFrame(new_results).reindex(columns=output_columns).to_csv(
                        OUTPUT_FILE, mode='a', header=False, index=False
                    )

    print("\n--- Simulation Complete ---")

if __name__ == '__main__':
    main()
