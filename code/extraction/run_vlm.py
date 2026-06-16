import os
import re
import json
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL.Image import Image as PILImage
from typing import NamedTuple, Optional
from pydantic import BaseModel

try:                                                  # newer vLLM
    from vllm.sampling_params import StructuredOutputsParams as _SO
    _SO_KW = "structured_outputs"
except ImportError:                                   # older vLLM
    from vllm.sampling_params import GuidedDecodingParams as _SO
    _SO_KW = "guided_decoding"


with open('../../prompts/relation_extraction_vlm.txt') as infile:
  EXTRACTION_PROMPT = infile.read()

PREDICTIONS_PATH  = "path_to_jsonl.jsonl"  # classification outputs
KEEP_LABELS       = ("research_model", "tested_model")
INPUT_DIR         = "path_to_mineru_papers"
BASE_DIR          = "path/to/in_context_samples"
LABELS_DIR        = "path/to/in_context_samples/labels"
CAPTION_DIR       = "path/to/in_context_samples"
IMGS_DIR          = "path/to/in_context_samples/images"
OUTDIR            = "output_directory"
OUT_FNAME         = "jsonl_output_file_name"
MODEL_NAME        = "google/gemma-4-31B-it"
MAX_TOKENS        = 8192
VISION_TOKENS     = 1120
FEW_SHOT_EXAMPLES = [
    "sample1.png",    # Example with moderators
    "sample2.png",    # Example with different sub-groups in one diagram
    "sample3.png",    # Example with multiple models in one image
    "sample4.png",    # Example without any edge labels
    "sample5.png",    # Example with * notation for significance levels
]

class ModelRequestData(NamedTuple):
    prompt: str
    image_data: list[PILImage]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def build_request(image_path, caption, processor):
    captions = _load_json(os.path.join(CAPTION_DIR, "captions.json"))
    prompt = EXTRACTION_PROMPT
    for i, ex_fname in enumerate(FEW_SHOT_EXAMPLES, start=1):
        stem = os.path.splitext(ex_fname)[0]
        with open(os.path.join(LABELS_DIR, f"{stem}.json")) as gtf:
            gt_str = gtf.read().strip()
        prompt = prompt.replace(f"<ground_truth{i}>", gt_str)

    images = [(os.path.join(IMGS_DIR, fn), captions.get(fn, "")) for fn in FEW_SHOT_EXAMPLES]
    images.append((image_path, caption))

    segments = prompt.split("<image>")

    content, image_paths = [], []
    for i, seg in enumerate(segments):
        if seg:
            content.append({"type": "text", "text": seg})
        if i < len(images):
            img_path_i, cap_i = images[i]
            content.append({"type": "image"})
            image_paths.append(img_path_i)
            if cap_i:
                content.append({"type": "text", "text": f'\nCaption: "{cap_i}"\n'})

    messages = [{"role": "user", "content": content}]
    prompt_str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return ModelRequestData(
        prompt=prompt_str,
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


def load_keep_ids(predictions_path, keep_labels=KEEP_LABELS):
    keep = set()
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label") not in keep_labels:
                continue
            img_field = rec["image"]
            img_id = img_field["image"] if isinstance(img_field, dict) else img_field
            keep.add(img_id)
    return keep


def get_batch(input_dir, processor, keep_ids):
    batch, ids = [], []
    for img_path, caption, img_id in iter_image_objects(input_dir):
        if img_id not in keep_ids:
            continue
        if not os.path.exists(img_path):
            print(f"[skip] missing image: {img_path}")
            continue
        req = build_request(img_path, caption, processor)
        batch.append({
            "prompt": req.prompt,
            "multi_modal_data": {"image": req.image_data},
            "mm_processor_kwargs": {"max_soft_tokens": VISION_TOKENS},
        })
        ids.append({"image": img_id, "caption": caption})
    return batch, ids


def parse_diagram(text):
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if isinstance(obj, list):
        obj = obj[0] if obj else None
    if obj is None:
        return None
    try:
        return Diagram(**obj).model_dump()
    except Exception:
        return obj


def run_extraction(llm, input_dir=INPUT_DIR, outdir=OUTDIR, outfname=OUT_FNAME):

    for run_i in range(1, 6):

        processor = AutoProcessor.from_pretrained(MODEL_NAME)

        keep_ids = load_keep_ids(PREDICTIONS_PATH)
        print(f"{len(keep_ids)} images classified as {KEEP_LABELS}")

        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=MAX_TOKENS,
            **{_SO_KW: _SO(json=RESPONSE_SCHEMA)},
        )

        batch, ids = get_batch(input_dir, processor, keep_ids)
        print(f"extracting from {len(ids)} images")

        outputs = llm.generate(batch, sampling_params)

        out_dir = os.path.join(outdir, "few_shot")
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, f"{outfname}_run{run_i}.jsonl")

        with open(results_path, "w") as f:
            for meta, output in zip(ids, outputs):
                raw = output.outputs[0].text
                f.write(json.dumps({
                    "image":   meta["image"],
                    "caption": meta["caption"],
                    "diagram": parse_diagram(raw),
                    "raw":     raw,
                }, ensure_ascii=False) + "\n")

        print(f"wrote {results_path}")

    return results_path


if __name__=='__main__':
  llm = LLM(model=MODEL_NAME,
            max_model_len=100000,  # due to computational ressources
            limit_mm_per_prompt={"image": 6})
  run_extraction(llm, input_dir=INPUT_DIR, outdir=OUTDIR, outfname=OUT_FNAME)
