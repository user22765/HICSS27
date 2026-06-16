import os
import re
import json
import vllm, torch
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from postprocessing import clean_relations, generate_feedback


MODEL_NAME       = "Qwen/Qwen3.5-27B"
N_TRIES          = 3
N_RUNS           = 5
N_REVISE         = 2
INPUT_DIR        = "../../data/val/mineru/val_pdfs"
OUTDIR           = "output_val_qwen"
MAX_TOKENS       = 16384
_THINK_RE        = re.compile(r'<think>.*?</think>', re.DOTALL)
PATH2PROMPT      = '../../prompts/relation_prompt.txt'
PATH2SCHEMA      = '../../prompts/relation_schema.txt'
PATH2CONSTPROMPT = '../../prompts/construct_prompt.txt'
PATH2CONSTSCHEMA = '../../prompts/construct_schema.txt'
os.makedirs(OUTDIR, exist_ok=True)

# Load LLM
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

print("vLLM:", vllm.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))

processor = AutoProcessor.from_pretrained(MODEL_NAME)

llm = LLM(
    model=MODEL_NAME,
    max_num_seqs=640,
)
print("vLLM engine initialized")


def _get_relations(parsed):
    return parsed['relations'] if isinstance(parsed, dict) and 'relations' in parsed else parsed


def _set_relations(parsed, rels):
    if isinstance(parsed, dict) and 'relations' in parsed:
        parsed['relations'] = rels
        return parsed
    return rels


def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def get_batch(processor, prompt_template, schema_md, input_dir=INPUT_DIR):
    mdfiles = sorted(os.path.join(input_dir, f, 'vlm', f'{f}.md') for f in os.listdir(input_dir))
    prompt_batch, user_texts = [], []
    for md_path in mdfiles:
        md_text   = re.sub(r'<details>.*?</details>', 
                           '', 
                           read_text(md_path), 
                           flags=re.DOTALL)
        full_text = f"{prompt_template}\n\n{schema_md}\n\n{md_text}"
        
        user_texts.append(full_text)
        messages = [{"role": "user", "content": [{"type": "text", "text": full_text}]}]
        prompt_batch.append({"prompt": processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)})
    return prompt_batch, mdfiles, user_texts


def build_revision_prompt(processor, user_text, current_relations, issues):
    numbered = json.dumps(current_relations, indent=2, ensure_ascii=False)
    feedback_block = (
        "The relations you extracted have the problems listed below. Return the COMPLETE "
        "corrected list as a JSON array inside a ```json block: include every relation, keep "
        "the already-correct ones unchanged and in the same order, and fix or remove only the "
        "flagged ones. Output JSON only, no commentary.\n\n"
        f"Relations you returned (0-indexed):\n{numbered}\n\n"
        "Problems to fix:\n- " + "\n- ".join(issues)
    )
    messages = [
        {"role": "user",      "content": [{"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": numbered}]},   # final answer only, no <think>
        {"role": "user",      "content": [{"type": "text", "text": feedback_block}]},
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def parse_json_block(text):
    text = _THINK_RE.sub('', text)
    m = (re.search(r'```json\s*(.*?)```', text, re.DOTALL) or re.search(r'```\s*(.*?)```', text, re.DOTALL))
    candidate = m.group(1) if m else text
    return json.loads(candidate.strip())


def run_generate(llm, 
                 path2prompt=PATH2PROMPT, 
                 path2schema=PATH2SCHEMA,
                 path2constprompt=PATH2CONSTPROMPT, 
                 path2constschema=PATH2CONSTSCHEMA, 
                 n_tries=N_TRIES,
                 input_dir=INPUT_DIR,
                 out_dir=OUTDIR,
                 temperature=0.5,
                 vision=False):
  
    processor         = AutoProcessor.from_pretrained(MODEL_NAME)
    prompt_template   = read_text(path2prompt)
    schema_md         = read_text(path2schema)
    construct_schema  = read_text(path2constprompt)
    extract_construct = read_text(path2constschema)
    extract_construct += f"\n\n{construct_schema}\n\nConstruct names:\n"
    t                 = temperature
    #out_dir           = OUTDIR
    os.makedirs(out_dir, exist_ok=True)

    for i in range(N_RUNS):
        extraction_prompts, md_names, user_texts = get_batch(processor, prompt_template, schema_md, input_dir)
        print(f"temp={t} | run {i+1}/{N_RUNS} | {len(md_names)} files")

        stems     = [os.path.splitext(f)[0].split('/')[-1] for f in md_names]
        out_paths = [os.path.join(out_dir, f"{s}_temp={t}_run{i+1}.json") for s in stems]
        sampling_params = SamplingParams(temperature=t, max_tokens=MAX_TOKENS)

        # Stage 1: Extract relations in valid json syntax
        print("     Extracting relations")
        results = {}
        pending = [k for k, p in enumerate(out_paths) if not os.path.exists(p)]
        # n_tries to generate valid outputs
        for attempt in range(1, n_tries + 1):
            if not pending:
                break
            # process batch
            outputs = llm.generate([extraction_prompts[k] for k in pending], sampling_params)
            still = []
            # iterate over generated extractions
            for k, output in zip(pending, outputs):
                raw = output.outputs[0].text
                try:
                    parsed = parse_json_block(raw)
                    if not parsed:
                        raise ValueError("empty result")
                    results[k] = parsed
                except Exception as e:
                    print(f"[WARN] {stems[k]}: parse attempt {attempt}/{n_tries} failed: {e}")
                    # re-process prompt if json was not parseable
                    still.append(k)
                    if attempt == n_tries:
                        with open(out_paths[k].replace('.json', '_FAILED-raw.txt'), 'w') as f:
                            f.write(raw)
            pending = still

        # Stage 2: validate and revise using check_valid feedback
        to_revise = {k: issues for k in results if (issues := generate_feedback(_get_relations(results[k])))}
        if len(to_revise) > 0:
            print("     Revising extracted relations:", len(to_revise), "extractions")
        for rev in range(1, N_REVISE + 1):
            if not to_revise:
                break
            ks = list(to_revise)
            rev_prompts = [build_revision_prompt(processor, user_texts[k],
                                                  _get_relations(results[k]), to_revise[k]) for k in ks]
            outputs = llm.generate(rev_prompts, sampling_params)
            next_round = {}
            for k, output in zip(ks, outputs):
                try:
                    new_rels   = _get_relations(parse_json_block(output.outputs[0].text))
                    new_issues = generate_feedback(new_rels)
                    # accept only on improvement
                    if len(new_issues) < len(to_revise[k]):
                        results[k] = _set_relations(results[k], new_rels)
                        if new_issues:
                            next_round[k] = new_issues
                    else:
                        # no progress -> retry
                        next_round[k] = to_revise[k]
                except Exception as e:
                    print(f"[WARN] {stems[k]}: revision parse failed: {e}")
                    next_round[k] = to_revise[k]
            to_revise = next_round

        # Apply post-processing routine
        for k in results:
            cleaned = clean_relations(_get_relations(results[k]), md_names[k], vision=vision)
            results[k] = _set_relations(results[k], cleaned)

        # Stage 3: extract construct definitions (same n_tries retry budget)
        print("     Extracting construct definitions and measurement items")
        construct_prompts = {}
        for k in results:
            rels = _get_relations(results[k])
            variables = []
            for x in rels:
                for key in ('cause', 'effect', 'mediator', 'moderator'):
                    v = x.get(key)
                    if v and v not in variables:
                        variables.append(v)
            paper_text = re.sub(r'<details>.*?</details>', '', read_text(md_names[k]), flags=re.DOTALL)
            user_text  = extract_construct + f"{variables}\n\n```{paper_text}```"
            msgs = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
            construct_prompts[k] = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        constructs = {}
        pending_c = list(construct_prompts)
        for attempt in range(1, n_tries + 1):
            if not pending_c:
                break
            outputs = llm.generate([construct_prompts[k] for k in pending_c], sampling_params)
            still = []
            for k, output in zip(pending_c, outputs):
                raw = output.outputs[0].text
                try:
                    parsed_c = parse_json_block(raw)
                    if not parsed_c:
                        raise ValueError("empty result")
                    constructs[k] = parsed_c
                except Exception as e:
                    print(f"[WARN] {stems[k]}: construct parse attempt {attempt}/{n_tries} failed: {e}")
                    still.append(k)
                    if attempt == n_tries:
                        with open(out_paths[k].replace('.json', '_constructs-FAILED-raw.txt'), 'w') as f:
                            f.write(raw)
            pending_c = still

        # write combined relations + constructs; flag anything imperfect
        for k, parsed in results.items():
            final = {'relations':  _get_relations(parsed),
                      'constructs': constructs.get(k, [])}
            with open(out_paths[k], 'w') as f:
                json.dump(final, f, indent=2)
            left = generate_feedback(final['relations'])
            if left:
                print(f"[INFO] {stems[k]}: saved with {len(left)} unresolved relation issue(s).")
            if k not in constructs:
                print(f"[INFO] {stems[k]}: saved WITHOUT constructs (extraction failed after {n_tries} tries).")

