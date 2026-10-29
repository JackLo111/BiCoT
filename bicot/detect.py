
import argparse
import json
import os

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

from bicot.data import get_gsm8k_dataset
from bicot.rsr_verifier import RSRBlackBoxVerifier


def partial_auc(fpr, tpr, max_fpr=0.05):
    stop = np.searchsorted(fpr, max_fpr, side="right")
    fpr_p, tpr_p = fpr[:stop], tpr[:stop]
    if len(fpr_p) > 0 and fpr_p[-1] < max_fpr:
        fpr_p = np.append(fpr_p, max_fpr)
        tpr_p = np.append(tpr_p, np.interp(max_fpr, fpr, tpr))
    return sklearn_auc(fpr_p, tpr_p) / max_fpr


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [x["question"] for x in json.load(f)]


@torch.no_grad()
def api_logits(model, tok, prompts, sentinels, max_new_tokens, device, bs=4):
    model.eval()
    out = []
    for i in range(0, len(prompts), bs):
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            )
            for q in prompts[i : i + bs]
        ]
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.pad_token_id,
        )
        steps = [s[:, sentinels].float().cpu().numpy() for s in gen.scores]
        out.append(np.mean(np.array(steps), axis=0))
    return np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--adapter_dir", required=True, help="dir with the LoRA adapter and secret_vector.pt"
    )
    ap.add_argument("--train_data", default="./cache/gsm8k_train_robust_v5.json")
    ap.add_argument("--test_data", default="./cache/gsm8k_test_robust_v5.json")
    ap.add_argument("--n_sentinels", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument(
        "--limit", type=int, default=0, help="use only the first N prompts per split (smoke test)"
    )
    ap.add_argument("--out", default="./outputs/detect_result.json")
    a = ap.parse_args()

    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if not (os.path.exists(a.train_data) and os.path.exists(a.test_data)):
        get_gsm8k_dataset(tok)
    train_q, test_q = load_questions(a.train_data), load_questions(a.test_data)
    if a.limit:
        train_q, test_q = train_q[: a.limit], test_q[: a.limit]
    n_cal = len(test_q) // 2
    calib, evalp, sent_prompts = test_q[:n_cal], test_q[n_cal:], train_q[: len(train_q) // 2]

    secret = torch.load(
        os.path.join(a.adapter_dir, "secret_vector.pt"), map_location="cpu", weights_only=True
    ).float()
    ver = RSRBlackBoxVerifier(tok.vocab_size, device=dev)

    base = AutoModelForCausalLM.from_pretrained(
        a.base_model, device_map=dev, torch_dtype=torch.float16
    )
    ver.select_optimal_sentinels(base, tok, secret, sent_prompts, n_sentinels=a.n_sentinels)
    ver.precompute_reference_statistics(base, tok, secret, calib)
    neg = ver.verify_detection(
        api_logits(base, tok, evalp, ver.sentinel_indices, a.max_new_tokens, dev), ver.w_sig
    )["sample_scores"]
    del base
    torch.cuda.empty_cache()

    wm = AutoModelForCausalLM.from_pretrained(
        a.base_model, device_map=dev, torch_dtype=torch.float16
    )
    wm = PeftModel.from_pretrained(wm, a.adapter_dir).merge_and_unload()
    w_rob = ver.robust_subspace_registration(
        api_logits(wm, tok, calib, ver.sentinel_indices, a.max_new_tokens, dev)
    )
    pos = ver.verify_detection(
        api_logits(wm, tok, evalp, ver.sentinel_indices, a.max_new_tokens, dev), w_rob
    )["sample_scores"]

    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    tpr_at = {f"TPR@FPR={f}": float(np.interp(f, fpr, tpr)) for f in (0.001, 0.01, 0.05, 0.1)}
    md = abs(pos.mean() - neg.mean()) / (np.sqrt((pos.std() ** 2 + neg.std() ** 2) / 2) + 1e-9)
    res = dict(
        n_pos=len(pos),
        n_neg=len(neg),
        mean_pos=float(pos.mean()),
        mean_neg=float(neg.mean()),
        auc=float(auc),
        pauc=float(partial_auc(fpr, tpr)),
        md=float(md),
        **tpr_at,
    )
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
