"""Weights: download + cache the public models, vendor+patch MOSS-TTS-Nano, and fetch the private
endpoint-head. All downloads go to HF_HOME (mounted cache); nothing is baked into the image.

MOSS-TTS-Nano patches (its trust_remote_code modeling is written for transformers 4.57; we also run it in
the MOUTH venv at 4.57, but these patches make it (a) load on either tf 4.57/5.x, (b) depend on soundfile
instead of torchcodec, (c) write NO audio to disk when output_audio_path is None (v1 privacy rule)):
  P1  _keys_to_ignore_on_load_unexpected: list -> None  (tf4.57 does list+, tf5.x does set|; None fits both)
  P2  torchaudio.load(ref)              -> soundfile.read     (drop torchcodec dep)
  P3  torchaudio.save(output_path, ...) -> soundfile.write, guarded by `output_audio_path is not None`
  P4  result yield audio_path           -> None when no path was given
Patches are idempotent (skipped if already applied)."""
from __future__ import annotations
import os, re, logging
from pathlib import Path

log = logging.getLogger("weights")


def _hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not t:
        tf = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "token"
        if tf.exists():
            t = tf.read_text().strip()
    return t or None


# ---- MOSS-TTS-Nano: snapshot + patch -----------------------------------------------------------
def ensure_moss(moss_repo: str, moss_dir: str) -> str:
    from huggingface_hub import snapshot_download
    snapshot_download(moss_repo, local_dir=moss_dir, token=_hf_token())
    _patch_moss_modeling(Path(moss_dir) / "modeling_moss_tts_nano.py")
    return moss_dir


def _patch_moss_modeling(mp: Path):
    s = mp.read_text(); orig = s

    # P1 ------------------------------------------------------------------
    s = s.replace('_keys_to_ignore_on_load_unexpected = [r"local_transformer\\.wte\\.weight"]',
                  '_keys_to_ignore_on_load_unexpected = None  # tyger: tf4.57 list+ / tf5.x set| both accept None')

    # P2 ------------------------------------------------------------------
    s = s.replace(
        'waveform, sample_rate = torchaudio.load(str(reference_audio_path))\n        waveform = waveform.to(torch.float32)',
        'import soundfile as _sf  # tyger: avoid torchcodec\n'
        '        _d, sample_rate = _sf.read(str(reference_audio_path), dtype="float32", always_2d=True)\n'
        '        waveform = torch.from_numpy(_d).to(torch.float32).transpose(0, 1).contiguous()')

    # P3 ------------------------------------------------------------------ (inference_stream save block)
    s = s.replace(
        '            decoded_sample_rate = decoded_sample_rate or target_sample_rate\n'
        '            output_path = Path(output_audio_path)\n'
        '            output_path.parent.mkdir(parents=True, exist_ok=True)\n'
        '            torchaudio.save(str(output_path), waveform, decoded_sample_rate)',
        '            decoded_sample_rate = decoded_sample_rate or target_sample_rate\n'
        '            if output_audio_path is not None:  # tyger: v1 writes NO audio to disk\n'
        '                output_path = Path(output_audio_path)\n'
        '                output_path.parent.mkdir(parents=True, exist_ok=True)\n'
        '                __import__("soundfile").write(str(output_path), torch.as_tensor(waveform).detach().cpu()'
        '.to(torch.float32).transpose(0, 1).contiguous().numpy(), int(decoded_sample_rate))')

    # P4 ------------------------------------------------------------------ (result yield audio_path when None)
    s = s.replace('                "audio_path": str(output_path),',
                  '                "audio_path": (str(output_audio_path) if output_audio_path is not None else None),')

    # remaining torchaudio.save (inference(), always has a path) -> soundfile
    s = s.replace(
        'torchaudio.save(str(output_path), waveform, decoded_sample_rate)',
        '__import__("soundfile").write(str(output_path), torch.as_tensor(waveform).detach().cpu()'
        '.to(torch.float32).transpose(0, 1).contiguous().numpy(), int(decoded_sample_rate))')

    if s != orig:
        mp.write_text(s)
        log.info("patched MOSS modeling at %s", mp)
    # sanity: the dangerous originals must be gone
    assert "torchaudio.load(str(reference_audio_path))" not in s, "P2 failed"
    assert "_keys_to_ignore_on_load_unexpected = [r" not in s, "P1 failed"


# ---- MOSS-TTS-Realtime: snapshot + static-cache/CUDA-graph patch -------------------------------
# The Realtime remote code (written for tf 5.0.0) decodes each audio frame through the Qwen3 backbone with
# a DynamicCache + growing attention mask and no cache_position -> shapes change every step, so the step
# can't be CUDA-graphed and runs at RTF ~1.6-2.0 on a 5090. We patch the Inference class to a bs=1 fast
# path: prefill/step through a preallocated StaticCache with explicit cache_position and a pure-causal mask,
# and run the single-token backbone step through torch.compile(mode="reduce-overhead") (CUDA graphs). With
# a long boot warmup to pre-capture the graph depth this holds a stable RTF ~0.55-0.7. Idempotent.
def ensure_moss_realtime(repo: str, moss_dir: str) -> str:
    from huggingface_hub import snapshot_download
    snapshot_download(repo, local_dir=moss_dir, token=_hf_token())
    _patch_realtime_streaming(Path(moss_dir) / "streaming_mossttsrealtime.py")
    return moss_dir


def _patch_realtime_streaming(sp: Path):
    s = sp.read_text()
    if "tyger: static-cache" in s:      # already patched (idempotent)
        return
    orig = s

    # E1: init fields + lazily-compiled backbone -------------------------------------------------
    s = s.replace(
        "        self._generated_tokens: List[torch.Tensor] = []\n"
        "        self._is_stopping = None\n"
        "        self._last_audio_tokens = None\n"
        "        self._step_idx = 0\n"
        "\n"
        "    @property\n"
        "    def device(self):",
        "        self._generated_tokens: List[torch.Tensor] = []\n"
        "        self._is_stopping = None\n"
        "        self._last_audio_tokens = None\n"
        "        self._step_idx = 0\n"
        "        # --- tyger: static-cache + CUDA-graph fast decode path (bs=1 only) ---\n"
        "        self._use_static = True\n"
        "        self._static_cache = None\n"
        "        self._cache_pos = 0\n"
        "        self._compiled_model = None\n"
        "\n"
        "    def _get_compiled_model(self):\n"
        "        if self._compiled_model is None:\n"
        "            self._compiled_model = torch.compile(self.model, mode=\"reduce-overhead\", fullgraph=False)\n"
        "        return self._compiled_model\n"
        "\n"
        "    @property\n"
        "    def device(self):")

    # E2: prefill into a StaticCache (bs=1) ------------------------------------------------------
    s = s.replace(
        "        outputs = self.model(\n"
        "            input_ids=current_input_ids,\n"
        "            attention_mask=current_attention_mask,\n"
        "            past_key_values=self.past_key_values,\n"
        "            use_cache=True,\n"
        "            return_dict=True,\n"
        "        )\n"
        "        self.past_key_values = outputs.past_key_values\n"
        "        self.attention_mask = current_attention_mask\n"
        "\n"
        "        backbone_hidden_states = outputs.last_hidden_state[:, -1:, :]",
        "        use_static = bool(getattr(self, \"_use_static\", False)) and current_input_ids.shape[0] == 1\n"
        "        if use_static:  # tyger: static-cache prefill (bs=1, pure-causal) for a CUDA-graphable step\n"
        "            if self._static_cache is None:\n"
        "                self._static_cache = StaticCache(config=self.model.language_model.config, max_cache_len=self.max_length)\n"
        "            else:\n"
        "                self._static_cache.reset()\n"
        "            self.past_key_values = self._static_cache\n"
        "            seqlen = current_input_ids.shape[1]\n"
        "            cache_position = torch.arange(seqlen, device=device)\n"
        "            outputs = self.model(\n"
        "                input_ids=current_input_ids, attention_mask=None,\n"
        "                past_key_values=self.past_key_values, use_cache=True, return_dict=True,\n"
        "                cache_position=cache_position,\n"
        "            )\n"
        "            self._cache_pos = seqlen\n"
        "        else:\n"
        "            outputs = self.model(\n"
        "                input_ids=current_input_ids, attention_mask=current_attention_mask,\n"
        "                past_key_values=self.past_key_values, use_cache=True, return_dict=True,\n"
        "            )\n"
        "        self.past_key_values = outputs.past_key_values\n"
        "        self.attention_mask = current_attention_mask\n"
        "\n"
        "        backbone_hidden_states = outputs.last_hidden_state[:, -1:, :].clone()")

    # E3: compiled single-token backbone step ----------------------------------------------------
    s = s.replace(
        "        outputs = self.model(\n"
        "            input_ids=step_ids,\n"
        "            attention_mask=self.attention_mask,\n"
        "            past_key_values=self.past_key_values,\n"
        "            use_cache=True,\n"
        "            return_dict=True,\n"
        "        )\n"
        "        self.past_key_values = outputs.past_key_values\n"
        "        backbone_hidden_states = outputs.last_hidden_state[:, -1:, :]",
        "        use_static = bool(getattr(self, \"_use_static\", False)) and self._static_cache is not None and batch_size == 1\n"
        "        if use_static:  # tyger: CUDA-graphed single-token backbone step (fixed shapes)\n"
        "            cache_position = torch.tensor([self._cache_pos], device=device, dtype=torch.long)\n"
        "            outputs = self._get_compiled_model()(\n"
        "                input_ids=step_ids, attention_mask=None, past_key_values=self._static_cache,\n"
        "                use_cache=True, return_dict=True, cache_position=cache_position,\n"
        "            )\n"
        "            self._cache_pos += 1\n"
        "        else:\n"
        "            outputs = self.model(\n"
        "                input_ids=step_ids, attention_mask=self.attention_mask,\n"
        "                past_key_values=self.past_key_values, use_cache=True, return_dict=True,\n"
        "            )\n"
        "        self.past_key_values = outputs.past_key_values\n"
        "        backbone_hidden_states = outputs.last_hidden_state[:, -1:, :].clone()")

    if s != orig:
        sp.write_text(s)
        log.info("patched MOSS-Realtime streaming at %s", sp)
    assert "tyger: static-cache" in s, "Realtime static-cache patch failed (E1 anchor not found)"
    assert s.count("tyger: CUDA-graphed single-token") == 1, "Realtime step patch failed (E3)"
    assert "static-cache prefill" in s, "Realtime prefill patch failed (E2)"


# ---- public models: warm the HF cache ----------------------------------------------------------
def ensure_public_cached(repos: list[str]):
    from huggingface_hub import snapshot_download
    for r in repos:
        snapshot_download(r, token=_hf_token())


# ---- private endpoint-head chain ---------------------------------------------------------------
def ensure_heads(repo: str, subpath: str, token: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    token = token or _hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN required for the private endpoint-head repo %r" % repo)
    path = hf_hub_download(repo_id=repo, filename=subpath, token=token)
    verify_head_ckpt(path)
    return path


def verify_head_ckpt(path: str):
    """Confirm the ep checkpoint carries the LoRA + all heads (so bc_s1 isn't needed separately)."""
    import torch
    ck = torch.load(path, map_location="cpu")
    need_top = {"base", "proj_agent", "proj_act", "yield_head", "bc_head", "ep_head"}
    missing = need_top - set(ck.keys())
    if missing:
        raise RuntimeError(f"endpoint ckpt {path} missing keys {missing}; may need bc_s1 too")
    base_keys = " ".join(ck["base"].keys())
    has_lora = "lora" in base_keys.lower()
    log.info("endpoint ckpt OK: heads=%s lora_in_base=%s", sorted(need_top), has_lora)
    if not has_lora:
        log.warning("no LoRA weights found inside ckpt['base'] — endpoint chain expects a LoRA backbone")
    return True
