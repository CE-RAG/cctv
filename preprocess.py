import json
import re
import difflib
from typing import Dict, List, Optional, Any, Set

def preprocess_query_to_english(
    text: str,
    thai_dict_path: str = "thai_dict.json",
    *,
    fuzzy_cutoff: float = 0.82,
    spell_cutoff: float = 0.72,
) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    has_thai = bool(re.search(r"[\u0E00-\u0E7F]", text))
    if not has_thai:
        return re.sub(r"\s+", " ", text).strip().lower()

    with open(thai_dict_path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)

    thai_map: Dict[str, str] = data.get("thai_map", {})
    if not thai_map:
        return ""

    categories = data.get("categories", {}) or {}
    colors: Set[str] = set(map(str.lower, categories.get("colors", [])))
    vehicle_types: Set[str] = set(map(str.lower, categories.get("vehicle_types", [])))
    generic_vehicle: Set[str] = set(map(str.lower, categories.get("generic_vehicle", ["car", "vehicle"])))

    keys: List[str] = list(thai_map.keys())
    keys_by_len = sorted(keys, key=len, reverse=True)

    def norm_thai(s: str) -> str:
        s = (s or "").strip()
        s = s.replace(" ", "").replace("-", "")
        try:
            import unicodedata
            s = unicodedata.normalize("NFC", s)
        except Exception:
            pass
        return s

    norm_key_to_key: Dict[str, str] = {}
    norm_keys: List[str] = []
    for k in keys:
        nk = norm_thai(k)
        if nk and nk not in norm_key_to_key:
            norm_key_to_key[nk] = k
            norm_keys.append(nk)

    def best_fuzzy_key(word: str) -> Optional[str]:
        m = difflib.get_close_matches(word, keys, n=1, cutoff=fuzzy_cutoff)
        return m[0] if m else None

    def spell_correct_thai(seg: str) -> Optional[str]:
        """
        คืนค่าเป็น 'key ภาษาไทยที่สะกดถูก' (ตามที่มีใน dict)
        """
        seg_n = norm_thai(seg)
        if not seg_n:
            return None


        if seg_n in norm_key_to_key:
            return norm_key_to_key[seg_n]

        cand = difflib.get_close_matches(seg_n, norm_keys, n=1, cutoff=spell_cutoff)
        if not cand:
            return None
        return norm_key_to_key.get(cand[0])

    def split_mixed(text_in: str) -> List[str]:
        return re.findall(r"[\u0E00-\u0E7F]+|[A-Za-z0-9]+", text_in)

    def longest_match(run: str) -> List[str]:
        out = []
        i = 0
        while i < len(run):
            for k in keys_by_len:
                if run.startswith(k, i):
                    out.append(k)
                    i += len(k)
                    break
            else:
                i += 1
        return out


    raw_tokens: List[str] = []
    for part in text.split():
        for seg in split_mixed(part):
            if re.search(r"[\u0E00-\u0E7F]", seg):
                corrected = spell_correct_thai(seg)

                if corrected:
                    raw_tokens.append(corrected)
                else:
                    raw_tokens.extend(longest_match(seg))
            else:
                raw_tokens.append(seg.lower())

    mapped: List[str] = []
    for tok in raw_tokens:
        if re.search(r"[\u0E00-\u0E7F]", tok):
            if tok in thai_map:
                v = (thai_map[tok] or "").strip().lower()
                if v:
                    mapped.append(v)
            else:
                best = best_fuzzy_key(tok)
                if best:
                    v = (thai_map[best] or "").strip().lower()
                    if v:
                        mapped.append(v)
        else:
            mapped.append(tok)

    if not mapped:
        return ""

    has_specific_type = any(t in vehicle_types for t in mapped)

    cleaned: List[str] = []
    for t in mapped:
        if has_specific_type and t in generic_vehicle:
            continue
        cleaned.append(t)

    color_tokens = [t for t in cleaned if t in colors]
    type_tokens = [t for t in cleaned if t in vehicle_types]
    other_tokens = [t for t in cleaned if t not in colors and t not in vehicle_types]

    final = color_tokens + other_tokens + type_tokens
    return re.sub(r"\s+", " ", " ".join(final)).strip()
