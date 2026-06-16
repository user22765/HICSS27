import os
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from enum import Enum
from PIL import Image
from PIL.Image import Image as PILImage
from typing import NamedTuple

# vLLM renamed the structured-output API across versions. Support both.
try:                                                  # newer vLLM
    from vllm.sampling_params import StructuredOutputsParams as _SO
    _SO_KW = "structured_outputs"
except ImportError:                                   # older vLLM
    from vllm.sampling_params import GuidedDecodingParams as _SO
    _SO_KW = "guided_decoding"


INPUT_DIR     = "path/to/mineru/outputs"
OUTDIR        = "outputs_sem_classification"
BASE_DIR      = "/content"
CAPTION_DIR   = "/content"
LABEL_DIR     = "/content"
MAX_TOKENS    = 512
VISION_TOKENS = 1120
os.makedirs(OUTDIR, exist_ok=True)


class Label(str, Enum):
    research_model = "research_model"
    tested_model   = "tested_model"
    other          = "other"

LABELS = [l.value for l in Label]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string", "maxLength": 300},
        "label":    {"type": "string", "enum": LABELS},
    },
    "required": ["evidence", "label"],
    "additionalProperties": False,
}

FEW_SHOT_EXAMPLES = [
    "example_conceptual.png",
    "example_results.png",
    "example_other.png",
]



SYSTEM_INSTRUCTIONS = f"""\
You are an expert in research methodology and structural equation modeling (SEM). You are shown a figure from an academic paper together with its caption. Classify the figure into EXACTLY ONE of these labels:

- "research_model": a CONCEPTUAL / hypothesized / proposed model that is intended to be tested with SEM. Latent constructs (boxes or ovals) are connected by directional arrows that represent hypotheses (often labelled H1, H2, ... or +/-). The arrows carry NO estimated numeric values, there are NO significance stars (*, **, ***) or p-values, and NO R-squared / variance-explained numbers inside the constructs.

- "tested_model": the structural model AFTER estimation. The arrows are annotated with estimated (standardized) path coefficients (e.g. 0.34, beta = 0.21), and/or there are significance markers (*, **, ***, or p-values), and/or R-squared / variance-explained values shown inside endogenous constructs. Solid vs. dashed paths may indicate supported vs. unsupported hypotheses.

- "other": anything that is not one of the above. Examples: a measurement model / CFA diagram showing only item loadings, a scatter/line/bar plot, a screenshot, a process flowchart unrelated to constructs, a table rendered as an image.

Decision rule (apply in order):
1. Primary evidence is VISUAL. The single most decisive cue is whether the PATHS between constructs carry estimated numbers / significance markers / R-squared values. * Numbers / stars / R-squared present on a construct-and-path diagram -> "tested_model". * A construct-and-path diagram with NO such estimates -> "research_model".
2. If the figure is not a construct-and-path SEM diagram at all -> "other".
3. The caption is SUPPORTING evidence only and can be misleading. Words like "research model", "conceptual model", "proposed model", "framework" lean toward research_model; "results", "estimated", "path coefficients", "tested" lean toward tested_model. If the caption contradicts the visual evidence, TRUST THE IMAGE.

Respond with a JSON object: a brief "evidence" field (one sentence naming the deciding cue) and a "label" field that is one of: {LABELS}. Output nothing else.
"""


class ModelRequestData(NamedTuple):
    prompt: str
    image_data: list[PILImage]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def build_request(image_path,
                  caption,
                  processor,
                  use_few_shot=True):

    captions = _load_json(os.path.join(CAPTION_DIR, "captions.json"))
    labels   = _load_json(os.path.join(LABEL_DIR, "labels.json")) if use_few_shot else {}

    content = [{"type": "text", "text": SYSTEM_INSTRUCTIONS}]
    image_paths = []

    if use_few_shot:
        content.append({"type": "text",
                        "text": "Here are labelled examples. Study them, then classify the final figure.\n"})
        for k, ex_fname in enumerate(FEW_SHOT_EXAMPLES, start=1):
            ex_path = os.path.join(BASE_DIR, ex_fname)
            image_paths.append(ex_path)
            content.append({"type": "text", "text": f"Example {k}:"})
            content.append({"type": "image"})
            content.append({"type": "text",
                            "text": f'Caption: "{captions.get(ex_fname, "")}"\n'
                                    f'Correct answer: {{"label": "{labels[ex_fname]}"}}\n'})

    content.append({"type": "text", "text": "Now classify THIS figure:"})
    content.append({"type": "image"})
    content.append({"type": "text",
                    "text": f'Caption: "{caption}"\n'
                            f'Respond with the JSON object only.'})
    image_paths.append(image_path)

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return ModelRequestData(
        prompt=prompt,
        image_data=[Image.open(p).convert("RGB") for p in image_paths],
    )


def join_caption(caption, footnote):
    parts = []
    for field in (caption, footnote):
        if isinstance(field, list):
            parts.extend(str(s).strip() for s in field if s)
        elif field:
            parts.append(str(field).strip())
    return " ".join(parts).strip()


def iter_image_objects(input_dir):
    md_names = sorted(
        os.path.join(input_dir, f, "vlm", f"{f}_content_list.json")
        for f in os.listdir(input_dir)
    )
    for fname in md_names:
        if not os.path.exists(fname):
            print(f"[skip] no content_list: {fname}")
            continue
        with open(fname) as infile:
            data = json.load(infile)
        pdf_id  = os.path.basename(fname).removesuffix("_content_list.json")
        vlm_dir = os.path.join(input_dir, pdf_id, "vlm")
        for img_obj in (x for x in data if x["type"] == "image"):
            caption = join_caption(img_obj.get("image_caption"),
                                   img_obj.get("image_footnote"))
            img_path = os.path.join(vlm_dir, img_obj["img_path"])
            yield img_path, caption, f"{pdf_id}/{img_obj['img_path']}"


def get_batch(input_dir, processor, use_few_shot=True):
    batch, ids = [], []
    for img_path, caption, img_id in iter_image_objects(input_dir):
        if not os.path.exists(img_path):
            print(f"[skip] missing image: {img_path}")
            continue
        req = build_request(img_path, caption, processor, use_few_shot=use_few_shot)
        batch.append({
            "prompt": req.prompt,
            "multi_modal_data": {"image": req.image_data},
            "mm_processor_kwargs": {"max_soft_tokens": VISION_TOKENS},
        })
        ids.append({"image": img_id, "caption": caption})
    return batch, ids


def parse_label(text):
    try:
        obj = json.loads(text)
        if obj.get("label") in LABELS:
            return obj["label"]
    except Exception:
        pass
    for lbl in LABELS:
        if lbl in text:
            return lbl
    return Label.other.value


def run_classification(llm, input_dir=INPUT_DIR, use_few_shot=True):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        **{_SO_KW: _SO(json=RESPONSE_SCHEMA)},
    )

    batch, image_names = get_batch(input_dir, processor, use_few_shot=use_few_shot)
    shot = "few_shot" if use_few_shot else "zero_shot"
    print(f"{shot} | {len(image_names)} images")

    outputs = llm.generate(batch, sampling_params)

    out_dir = os.path.join(OUTDIR, shot)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "pubmed_test_predictions.jsonl")

    with open(results_path, "w") as f:
        for fname, output in zip(image_names, outputs):
            raw = output.outputs[0].text
            f.write(json.dumps({
                "image": fname,
                "label": parse_label(raw),
                "raw":   raw,
            }) + "\n")

    return results_path


if __name__=='__main__':
    llm = LLM(model=MODEL_NAME,
          max_model_len=100000,  # due to computational ressources
          limit_mm_per_prompt={"image": 6})
    run_classification(llm)
