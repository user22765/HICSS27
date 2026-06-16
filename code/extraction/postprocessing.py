import re, os, json
from tqdm import tqdm
import xml.etree.ElementTree as ET
from abbreviations import schwartz_hearst


def filter_constructs(rels, constructs):
    var_names = set()
    for r in rels:
        for field in ("cause", "effect", "moderator", "mediator"):
            v = r.get(field)
            if v:
                var_names.add(v)
    
    return [c for c in constructs if c['name'] in var_names]

# Delimiters that signal a composite: " x ", "×", ", ", " and "
_SPLIT = re.compile(r"\s+x\s+|\s*×\s*|\s*,\s*|\s+and\s+")

def filter_relations_without_beta(relations, verbose=False):
    with_beta = [r for r in relations if r.get("beta") is not None]
    if not with_beta:
        if verbose and relations:
            print(f"[beta filter] no relation has a beta — keeping all "
                  f"{len(relations)} relations")
        return relations
    removed = len(relations) - len(with_beta)
    if verbose and removed:
        for r in relations:
            if r.get("beta") is None:
                print(f"[beta filter] removed: {r.get('cause')} -> "
                      f"{r.get('effect')} (status: {r.get('status')})")
    return with_beta


def _variable_names(relations):
    """All variable names that appear anywhere in the relation set."""
    names = set()
    for r in relations:
        for field in ("cause", "effect", "moderator"):
            v = r.get(field)
            if v:
                names.add(v.strip().lower())
    return names


def _is_composite(name, names):
    """True if `name` splits into two or more parts, all known variables."""
    parts = [p.strip().lower() for p in _SPLIT.split(name) if p.strip()]
    return len(parts) > 1 and all(p in names for p in parts)


def remove_composite_relations(relations):
    """
    Drop every relation whose cause, effect, or moderator is a composite
    of two other variables in the set. Returns the filtered list.
    """
    names = _variable_names(relations)
    return [
        r for r in relations
        if not any(
            f and _is_composite(f, names)
            for f in (r.get("cause"), r.get("effect"), r.get("moderator"))
        )
    ]


def transform_vision_relations(rels):
    
    keep_rels = [] 
    for rel in rels:

        transf_rel = {'cause': rel['cause'],
                      'effect': rel['effect'],
                      'moderator': rel['moderator'],
                      'beta': rel['path_coefficient'],
                      'p': rel['p_value'],
                      'hypothesis_id': rel['hyp_id'],
                      'model_id': rel['model_id']}
        
        if transf_rel['beta']:
            keep_rels.append(transf_rel)
    
    if len(keep_rels) == 0:
        for rel in rels:
            transf_rel = {'cause': rel['cause'],
                          'effect': rel['effect'],
                          'moderator': rel['moderator'],
                          'beta': rel['path_coefficient'],
                          'p': rel['p_value'],
                          'hypothesis_id': rel['hyp_id'],
                          'model_id': rel['model_id']}
            
            if transf_rel['hypothesis_id']:
                keep_rels.append(transf_rel)
    
    return keep_rels


def clean_relations(rels):
    # 1. model_id mirroring hypothesis_id -> default main model
    for r in rels:
        hyp = r.get('hypothesis_id')
        if hyp is not None and r.get('model_id') == hyp:
            r['model_id'] = "Main Model (default)"

    # 2. dedup on (cause, effect, moderator, beta, p)
    kept = {}
    for r in rels:
        key = (r.get('cause'), r.get('effect'), r.get('moderator'),
               r.get('beta'), r.get('p'))
        if key not in kept:
            kept[key] = r
            continue
        keeper = kept[key]
        for field, value in r.items():
            if value in (None, "") or keeper.get(field) not in (None, ""):
                continue
            keeper[field] = value
    rels = list(kept.values())

    # 3. within (cause, effect, moderator): if the relation exists both
    #    without beta and with beta, drop the beta-less one(s) and hand
    #    their hypothesis_id to the with-beta relation(s) that lack it
    def has_beta(r):
        return r.get('beta') not in (None, "")

    groups = {}
    for r in rels:
        groups.setdefault(
            (r.get('cause'), r.get('effect'), r.get('moderator')), []
        ).append(r)

    remove_ids = set()
    for group in groups.values():
        with_beta = [r for r in group if has_beta(r)]
        without_beta = [r for r in group if not has_beta(r)]
        if not with_beta or not without_beta:
            continue  # need both present to act

        # take a hypothesis_id from the beta-less relation(s) being dropped
        donor_hyp = next(
            (r.get('hypothesis_id') for r in without_beta
             if r.get('hypothesis_id') not in (None, "")),
            None,
        )
        if donor_hyp is not None:
            for r in with_beta:
                if r.get('hypothesis_id') in (None, ""):
                    r['hypothesis_id'] = donor_hyp

        for r in without_beta:
            remove_ids.add(id(r))

    return [r for r in rels if id(r) not in remove_ids]


def remove_indicators(relations, print_res=False, p=True, remove_abbrev=False):
    INDICATOR_RE    = re.compile(r"\b[A-Za-z]{1,6}[ _.\-]?\d{1,3}[A-Za-z]?\b")
    ALLCAPS_CODE_RE = re.compile(r"[A-Z]+\d{1,3}")
    cleaned         = []
    removed         = []

    for rel in relations:
        cause  = rel.get("cause", "") or ""
        effect = rel.get("effect", "") or ""
        mod    = rel.get("moderator", "") or ""
        p_val  = rel.get("p_value", "") or ""
        cause  = cause.replace('_', '').replace('-', '').strip()
        effect = effect.replace('_', '').replace('-', '').strip() 

        if not cause or not effect:
            continue
        if remove_abbrev and (len(cause) <= 2 or len(effect) <= 2):
            continue
        if (bool(INDICATOR_RE.fullmatch(cause.strip())) or bool(ALLCAPS_CODE_RE.fullmatch(cause.strip()))) and (not p_val or not p):
            removed.append(cause)
            continue
        if (bool(INDICATOR_RE.fullmatch(effect.strip())) or bool(ALLCAPS_CODE_RE.fullmatch(effect.strip()))) and (not p_val or not p):
            removed.append(effect)
            continue
        if (bool(INDICATOR_RE.fullmatch(mod.strip())) or bool(ALLCAPS_CODE_RE.fullmatch(mod.strip()))) and (not p_val or not p):
            removed.append(mod)
            continue
        cleaned.append(rel)
    
    if removed and print_res:
        print(removed)
        #print(json.dumps(cleaned, indent=2))
    return cleaned, removed


def remove_indicators_image(relations, constructs):

    def _matches(s):
        return '_' in s or s.islower() or s.isupper() or len(s) < 4

    valid = []
    for rel in relations:
        print(rel)
        # guard: skip endpoints with digit-ending or very short names
        if (rel['cause'][-1].isdigit() or len(rel['cause']) <= 2) or \
           (rel['effect'][-1].isdigit() or len(rel['effect']) <= 2):
            continue

        cause_item = [x.get('is_item') for x in constructs if x['name'] == rel['cause']]
        effect_item = [x.get('is_item') for x in constructs if x['name'] == rel['effect']]

        # flag: an item-construct with a suspicious name
        if cause_item and cause_item[0] and _matches(rel['cause']) and len(rel['cause']) < 10:
            continue
        if effect_item and effect_item[0] and _matches(rel['effect']) and len(rel['effect']) < 10:
            continue

        valid.append(rel)
    return valid


def extract_json(path2file):

    with open(path2file) as infile:
        model_output = infile.read()

    json_txt  = model_output.split('```json')[1].split('```')[0]
    return json.loads(json_txt)


def format_pvalues(preds):

    p_pattern = re.compile(r"(p|_p_|_p|p_)?\s*(<=|>=|<|>|=|≥|≤)?\s*(\d?\.\d{1,4})",
                           re.IGNORECASE)

    for i, rel in enumerate(preds):

        if 'p' in rel.keys():

            if rel['p']:

                rel['p'] = str(rel['p'])

                res = p_pattern.findall(rel['p'])

                if rel['p'] == '0':
                    preds[i]['p_unformatted'] = rel['p']
                    preds[i]['p'] = '< 0.001'
                    continue

                if not res:
                    preds[i]['p_unformatted'] = rel['p']
                    preds[i]['p'] = None
                    continue

                operator, val = res[0][1], res[0][2]

                if (operator == '<') or (operator == '<=') or (operator == '≤') or (operator == '') or (operator == '='):
                    preds[i]['p_unformatted'] = rel['p']
                    val = float(val)
                    if val <= 0.001:
                        preds[i]['p'] = '< 0.001'
                    elif val <= 0.01:
                        preds[i]['p'] = '< 0.01'
                    elif val <= 0.05:
                        preds[i]['p'] = '< 0.05'
                    elif val <= 0.1:
                        preds[i]['p'] = '< 0.1'
                    else:
                        preds[i]['p'] = '> 0.1'

                elif (operator == '>')  or (operator == '>=') or (operator == '≥'):
                    preds[i]['p_unformatted'] = rel['p']
                    if float(val) >= 0.1:
                        preds[i]['p'] = '> 0.1'
                    else:
                        preds[i]['p'] = None

            else:
                preds[i]['p'] = None
                preds[i]['p_unformatted'] = None
        else:

            preds[i]['p'] = None
            preds[i]['p_unformatted'] = None

    return preds


def filter_valid(preds):

    valid_rels = []

    for rel in preds:
        x, y = rel.get('cause'), rel.get('effect')
        med, mod = rel.get('mediator', ''), rel.get('moderator', '')

        # Check if cause, effect, moderator, beta and p exist
        cond0 = 'cause' in rel.keys() and 'effect' in rel.keys() and 'beta' in rel.keys() and 'p' in rel.keys()
        # Check if cause or effect are same or non existing
        cond1 = x != y and x is not None and y is not None
        # Check if cause, effect, mediator, moderator are of type string or None
        cond2 = type(x) == str and type(y) == str and (type(med) == str or med is None) and (type(mod) == str or mod is None)
        # Check if cause = moderator/mediator or effect = moderator/mediator
        cond3 = not (x == med or x == mod or y == med or y == mod)

        if cond0 and cond1 and cond2 and cond3:
            valid_rels.append(rel)

    return valid_rels


def generate_feedback(preds):
    feedback = []

    for i, rel in enumerate(preds):
        if not isinstance(rel, dict):
            feedback.append(f"Relation #{i} is not a JSON object; each relation must be an "
                            f"object with 'cause', 'effect', 'beta' and 'p' fields.")
            continue

        x, y = rel.get('cause'), rel.get('effect')
        med, mod = rel.get('mediator'), rel.get('moderator')
        label = f"Relation #{i} (cause={x!r} -> effect={y!r})"

        # 1. required fields must be present  (was cond0)
        for field in ('cause', 'effect', 'beta', 'p'):
            if field not in rel:
                hint = (" (use null only if the paper genuinely does not report it)."
                        if field in ('beta', 'p')
                        else " with the construct name as a string.")
                feedback.append(f"{label}: missing required field '{field}'. Add it{hint}")

        # 2. cause / effect must have actual values  (was part of cond1)
        if 'cause' in rel and x is None:
            feedback.append(f"{label}: 'cause' is null. Every relation needs a named cause construct.")
        if 'effect' in rel and y is None:
            feedback.append(f"{label}: 'effect' is null. Every relation needs a named effect construct.")

        # 3. cause and effect must differ  (was part of cond1)
        if x is not None and y is not None and x == y:
            feedback.append(f"{label}: 'cause' and 'effect' are identical ({x!r}). A relation must "
                            f"connect two different constructs — correct the names or drop it.")

        # 4. cause / effect must be strings  (was part of cond2)
        if x is not None and not isinstance(x, str):
            feedback.append(f"{label}: 'cause' must be a text string, but got {type(x).__name__}.")
        if y is not None and not isinstance(y, str):
            feedback.append(f"{label}: 'effect' must be a text string, but got {type(y).__name__}.")

        # 5. mediator / moderator must be string or null  (was part of cond2)
        if med is not None and not isinstance(med, str):
            feedback.append(f"{label}: 'mediator' must be a string or null, but got {type(med).__name__}.")
        if mod is not None and not isinstance(mod, str):
            feedback.append(f"{label}: 'moderator' must be a string or null, but got {type(mod).__name__}.")

        # 6. a construct cannot play two roles  (was cond3)
        if med:
            if med == x:
                feedback.append(f"{label}: 'mediator' ({med!r}) equals the cause. A construct cannot be both.")
            if med == y:
                feedback.append(f"{label}: 'mediator' ({med!r}) equals the effect. A construct cannot be both.")
        if mod:
            if mod == x:
                feedback.append(f"{label}: 'moderator' ({mod!r}) equals the cause. A construct cannot be both.")
            if mod == y:
                feedback.append(f"{label}: 'moderator' ({mod!r}) equals the effect. A construct cannot be both.")

    return feedback


def get_paper_abbreviations(path2file):

    abbreviations = {}

    if path2file.endswith('.grobid.tei.xml'):
        if os.path.exists(path2file):
            tree = ET.parse(path2file)
            xml_str = ET.tostring(tree.getroot(),
                                encoding='utf-8',
                                method='text')
            xml_str = xml_str.decode('utf-8')
            texts = xml_str.split('. ')

    elif path2file.endswith('.md'):
        if os.path.exists(path2file):
            with open(path2file) as infile:
                txt_str = infile.read()
                texts = txt_str.split('. ')
    else:
        return {}

    for t in texts:

        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=t,
                                                                      most_common_definition=True)

        for k_pairs, v_pairs in pairs.items():

            if not k_pairs in abbreviations.keys():
                abbreviations[k_pairs] = v_pairs

    return abbreviations


def get_abbreviation_map_vars(var_preds):

    abbreviations = {}

    for t in var_preds:

        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=t,
                                                                      most_common_definition=True)

        for k, v in pairs.items():

            if not k in abbreviations.keys():
                abbreviations[k] = v

    return abbreviations


def clean_names(preds, abbreviations):

    for i, rel in enumerate(preds):
        if rel['cause'] in abbreviations:
            preds[i]['cause'] = abbreviations[rel['cause']]
        if rel['effect'] in abbreviations:
            preds[i]['effect'] = abbreviations[rel['effect']]
        if 'moderator' in rel.keys():
            if rel['moderator'] in abbreviations:
                preds[i]['moderator'] = abbreviations[rel['moderator']]
        else:
            preds[i]['moderator'] = None
        if 'mediator' in rel.keys():
            if rel['mediator'] in abbreviations:
                preds[i]['mediator'] = abbreviations[rel['mediator']]
        else:
            preds[i]['mediator'] = None

    text_vars = set([t['cause'] for t in preds] + [t['effect'] for t in preds] + [t['mediator'] for t in preds if 'mediator' in t.keys()] + [t['moderator'] for t in preds if 'moderator' in t.keys()])

    mapping_dict = get_abbreviation_map_vars(text_vars)

    for k, v in mapping_dict.items():

        for idx, rel in enumerate(preds):

            if f'({k.lower()})' in rel['cause'].lower():
                preds[idx]['cause'] = preds[idx]['cause'].replace(f'({k})', '').strip()
            if f'({k.lower()})' in rel['effect'].lower():
                preds[idx]['effect'] = preds[idx]['effect'].replace(f'({k})', '').strip()
            if 'moderator' in rel.keys() and rel['moderator']:
                if f'({k.lower()})' in rel['moderator'].lower():
                    preds[idx]['moderator'] = preds[idx]['moderator'].replace(f'({k})', '').strip()
            if 'mediator' in rel.keys() and rel['mediator']:
                if f'({k.lower()})' in rel['mediator'].lower():
                    preds[idx]['mediator'] = preds[idx]['mediator'].replace(f'({k})', '').strip()

    not_printed = True

    for k, v in abbreviations.items():
        if k.lower() in [x.lower() for x in text_vars if x]:
            if not_printed:
                not_printed = False

    return preds


def remove_additional_hypotheses(relations):
    remove_ids = set()

    for rel in relations:
        # skip relations that already have a path coefficient
        if rel.get("path_coefficient") not in (None, "") and rel.get("p_value") not in (None, ""):
            continue

        # rel is missing path_coefficient + p_value
        hyp_id = rel.get("hyp_id") or None
        key = (rel.get("cause"), rel.get("effect"), rel.get("moderator"))

        # find the same relation that DOES have a path coefficient
        matches = [
            r for r in relations
            if id(r) != id(rel)
            and (r.get("cause"), r.get("effect"), r.get("moderator")) == key
            and r.get("path_coefficient") not in (None, "")
            and r.get("p_value") not in (None, "")
        ]

        if matches:
            if hyp_id is not None:
                for m in matches:
                    m["hyp_id"] = hyp_id
            #print(rel)
            remove_ids.add(id(rel))

    return [r for r in relations if id(r) not in remove_ids]



_CANON = [0.001, 0.01, 0.05, 0.1]
_STAR  = r'\\?\*'                              # one star, optionally escaped: * or \*
_STARS = rf'(?:{_STAR})+'                      # a run: *, **, ***, \*\*, ...
_GAP   = r'[\s:.;,\-–—]*'                      # separators between stars and the statement
_VERB  = r'(?:indicat\w+|denot\w+|repres\w+|mean\w+|signif\w*|stand\w*\s+for)'
_P     = r'[pP](?:[\s.\-]*(?:value|val))?'     # p, P, p-value, p val, p-val
_OP    = r'(?:<=?|≤|⩽|=|>)'                    # <, <=, ≤, =, >
_NUM2   = r'(?P<num>0?\.\d+|\d+(?:\.\d+)?\s*%|\d+\s*(?:percent|per\s*cent))'
_CLAUSE = (
    rf'(?:'
      rf'(?:{_P}\s*{_OP}\s*|significan\w*\s+at\s+(?:the\s+)?){_NUM2}(?:\s*(?:level|significance))?'
      rf'|(?:not\s+signif\w+|non[\s-]?signif\w+|n\.\s?s\.?)'
    rf')'
)
DEFINITION_RE2 = re.compile(
    rf'(?<![\w.*\\])(?P<stars>{_STARS}){_GAP}(?:{_VERB}{_GAP})?{_CLAUSE}',
    re.IGNORECASE,
)


def _to_level(num_str):
    s = num_str.strip().lower()
    digits = float(re.sub(r'[^\d.]', '', s))
    val = digits / 100.0 if ('%' in s or 'percent' in s or 'per cent' in s) else digits
    fits = [c for c in _CANON if c >= val - 1e-12]   # tightest standard level the value satisfies
    chosen = min(fits) if fits else 0.1              # values > 0.1 fall back to the loosest
    return f"<{'%g' % chosen}"


def build_significance_map(path2file):
    with open(path2file) as infile:
        text = infile.read()

    mapping = {}
    for m in DEFINITION_RE2.finditer(text):
        key = m.group('stars').replace('\\', '').strip()   # \*\* -> **
        num = m.group('num')
        mapping[key] = _to_level(num) if num else 'n.s.'
    return mapping


def map_significance(rels, path2text):
    # no relation contains significance levels in * notation
    if any([not 'p' in rel.keys() for rel in rels]):
        return rels
    if not any(['*' in str(rel['p']) for rel in rels]):
        return rels
    # get map
    star_map = build_significance_map(path2text)
    # check wether map is empty
    if len(star_map.keys()) == 0:
        return rels
    # map stars to significance levels
    updated_rels = []
    for rel in rels:
        if not rel['p']:
            updated_rels.append(rel)
            continue
        if '*' in str(rel['p']):
            sig_str = '*'*sum([1 for x in str(rel['p']) if x=='*'])
            if rel['p'] in star_map.keys():
                print(rel['p'], '->', star_map[sig_str])
                rel['p'] = star_map[sig_str]
        updated_rels.append(rel)
    return updated_rels


def check_betas(rels, path2file):
    
    results = []

    with open(path2file) as infile:
        text = infile.read()

    for rel in rels:
        beta = rel['beta']
        if beta and type(beta) in [int, float]:
            if beta < 0:
                beta1, beta2 = str(beta), '-' + str(beta)[2:]
            else:
                beta1, beta2 = str(beta), str(beta)[1:]
            if not (beta1 in text or beta2 in text):
                print(path2file.split('/')[-1], f"Didn't find beta={beta} in the text")
                rel['beta'] = None
        else:
            rel['beta'] = None
        results.append(rel)

    return results


def map_variables(rels, equivs):
    result = []
    for rel in rels:
        if rel.get('cause', '') in equivs.keys():
            rel['cause'] = equivs[rel['cause']]
        if rel.get('effect', '') in equivs.keys():
            rel['effect'] = equivs[rel['effect']]
        if rel.get('moderator', '') in equivs.keys():
            rel['moderator'] = equivs[rel['moderator']]
        if rel.get('mediator', '') in equivs.keys():
            rel['mediator'] = equivs[rel['mediator']]
        result.append(rel)
    return result


def clean_relations(rels, constructs, path2file, vision=False, file_equivs=None, drop_missing_beta=False):
    abbreviations = get_paper_abbreviations(path2file)
    rels, _ = remove_indicators(rels)
    if vision:
        rels = transform_vision_relations(rels)
    rels = map_significance(rels, path2file)
    rels = filter_valid(rels)
    rels = format_pvalues(rels)
    rels = clean_names(rels, abbreviations)
    rels, _ = remove_indicators(rels, p=False, remove_abbrev=True)
    if vision:
        rels = remove_indicators_image(rels, constructs)
    if not vision: 
        rels = check_betas(rels, path2file)
    if file_equivs:
        rels = map_variables(rels, file_equivs)
    curr_len = len(rels)
    rels = remove_composite_relations(rels)
    if len(rels) != curr_len:
        print("[REMOVED COMPOSITES]")
    if drop_missing_beta:
        rels = filter_relations_without_beta(rels, verbose=False)
    return rels


def clean_predictions(base_dir_pred, 
                      engine_suffix, 
                      files, 
                      base_dir_md,
                      vision=False, 
                      save_results=True,
                      equiv_json=None,
                      file_suffix=None,
                      drop_missing_beta=False,
                      out_suffix='_cleaned.json'):
    
    if equiv_json:
        with open(equiv_json) as f:
            equivs_map = json.load(f)
    else:
        equivs_map = {}

    for fname in tqdm(sorted(files)):
        if '_cleaned' in fname or not fname.endswith('.json'):
            continue

        with open(os.path.join(base_dir_pred, fname)) as f:
            model_output = json.load(f)

        base_name = fname.replace('.json', '')
        if file_suffix:
            paper_id  = base_name[:-len(file_suffix)]
        else:  # default _runX
            paper_id  = base_name[:-5]

        path2file = os.path.join(base_dir_md, paper_id, 'vlm', paper_id + engine_suffix)
        rels = model_output['relations']

        try:
            if equivs_map and paper_id in equivs_map.keys():
                file_equivs = equivs_map[paper_id]
            else:
                file_equivs = None
            rels = clean_relations(rels, model_output['constructs'], path2file, vision=vision, file_equivs=file_equivs, drop_missing_beta=drop_missing_beta)
            cons = filter_constructs(rels, model_output['constructs'])
        except Exception as e:
            print("[INFO] Cleaning failed:", fname)
            print(e)
            rels = model_output['relations']
        model_output['relations'] = rels
        model_output['constructs'] = cons

        if save_results:
            with open(os.path.join(base_dir_pred, base_name + out_suffix), 'w') as outfile:
                json.dump(model_output, outfile, indent=2)
