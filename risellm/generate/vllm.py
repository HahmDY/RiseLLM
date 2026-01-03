import os
from copy import deepcopy
from multiprocessing import Process, Manager, set_start_method
from typing import List, Union, Dict, Any


def _parse_gpus(gpus_arg: str) -> List[int]:
    parts = [p.strip() for p in gpus_arg.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _shard_indices(total: int, rank: int, world: int) -> List[int]:
    # Deterministic sharding: rank r handles indices i where i % world == r
    return [i for i in range(total) if i % world == rank]


def _worker_generate(
    gpu_id: int,
    rank: int,
    world: int,
    model_name: str,
    prompts: List[List[Dict[str, Any]]],
    shared_out,  # Manager().dict() proxy
    chat: bool,
    do_sample: bool,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    responses: int,
):
    # Bind this process to a single GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import inside the worker to avoid CUDA context issues in the parent process.
    from risellm.llm import VLLM

    llm = VLLM(model_name=model_name)
    indices = _shard_indices(len(prompts), rank, world)

    # If do_sample is False, force deterministic settings.
    eff_temperature = temperature if do_sample else 0.0
    eff_top_p = top_p if do_sample else 1.0

    for i in indices:
        msg = deepcopy(prompts[i])

        # If chat=True, prepend an empty system message (same convention as your earlier script).
        if chat:
            msg = [{"role": "system", "content": ""}] + msg

        # Try to generate multiple responses in one call if the wrapper supports it.
        # Otherwise, fall back to repeated single generations.
        outs: List[str] = []
        try:
            out = llm.generate(
                msg,
                temperature=eff_temperature,
                max_new_tokens=max_new_tokens,
                top_p=eff_top_p,
                n=responses,  # may not be supported by your wrapper
            )

            # Normalize to a list of length `responses`.
            if isinstance(out, list):
                outs = out
            else:
                outs = [out]

        except TypeError:
            # Wrapper does not support n=...
            for _ in range(responses):
                outs.append(
                    llm.generate(
                        msg,
                        temperature=eff_temperature,
                        max_new_tokens=max_new_tokens,
                        top_p=eff_top_p,
                    )
                )

        # Ensure exact length `responses` (pad by repeating last / trim if needed).
        if len(outs) < responses:
            pad_val = outs[-1] if len(outs) > 0 else ""
            outs.extend([pad_val] * (responses - len(outs)))
        else:
            outs = outs[:responses]

        shared_out[i] = outs


def generate_vllm(
    model_name,
    messages,
    chat=True,
    do_sample=True,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=1.0,
    gpus="0,1,2,3",
    responses=10,
):
    """
    Parallel vLLM generation over multiple GPUs.

    Args:
        model_name: HuggingFace model name or path.
        messages:
            - Either List[List[{"role":..., "content":...}]] (multiple prompts)
            - Or List[{"role":..., "content":...}] (single prompt)
        chat: If True, prepend {"role":"system","content":""}.
        do_sample: If False, forces temperature=0 and top_p=1.
        temperature: Sampling temperature when do_sample=True.
        max_new_tokens: Max tokens to generate.
        top_p: Nucleus sampling p when do_sample=True.
        gpus: Comma-separated GPU ids, e.g. "0,1,2,3".
        responses: Number of responses per prompt.

    Returns:
        A nested list shaped like:
            [
              [resp_0_for_prompt0, resp_1_for_prompt0, ..., resp_{responses-1}_for_prompt0],
              [resp_0_for_prompt1, resp_1_for_prompt1, ..., resp_{responses-1}_for_prompt1],
              ...
            ]
    """
    gpu_list = _parse_gpus(gpus)
    if not gpu_list:
        raise ValueError("No GPUs provided. Use gpus like '0,1,2,3'.")

    # Normalize input to List[List[dict]].
    if messages and isinstance(messages, list) and isinstance(messages[0], dict) and "role" in messages[0]:
        prompts = [messages]  # single prompt
    else:
        prompts = messages  # assume already list of prompts

    if not isinstance(prompts, list) or (len(prompts) > 0 and not isinstance(prompts[0], list)):
        raise TypeError(
            "messages must be either a single prompt (List[dict]) or multiple prompts (List[List[dict]])."
        )

    world = len(gpu_list)

    try:
        set_start_method("spawn")
    except RuntimeError:
        # Start method already set in this Python process.
        pass

    with Manager() as manager:
        shared_out = manager.dict()  # index -> List[str]

        procs: List[Process] = []
        for rank, gpu_id in enumerate(gpu_list):
            p = Process(
                target=_worker_generate,
                kwargs=dict(
                    gpu_id=gpu_id,
                    rank=rank,
                    world=world,
                    model_name=model_name,
                    prompts=prompts,
                    shared_out=shared_out,
                    chat=chat,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    responses=responses,
                ),
                daemon=False,
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # Collect outputs in original order.
        out: List[List[str]] = []
        for i in range(len(prompts)):
            # If something failed, return an empty list for that prompt index.
            out.append(list(shared_out.get(i, [""] * responses)))

        return out
