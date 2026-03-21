# blip_encoder.py
# BLIP-1 semantic embedding encoder — Python 3.7, transformers==4.30.2
#
# FIX: In transformers 4.30.2, BlipForConditionalGeneration does NOT
# have a 'text_encoder' attribute. The BERT backbone is accessed via:
#   model.text_decoder.bert
#
# Pipeline:
#   1. ViT image encoder        -> image features
#   2. BLIP text_decoder        -> generate caption
#   3. Tokenizer                -> token ids
#   4. text_decoder.bert        -> contextual hidden states (seq x 768)
#   5. Mean pool + L2 norm      -> 768-dim float32 embedding

import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from parameters import (
    BLIP_MODEL_NAME,
    BLIP_EMBEDDING_DIM,
    BLIP_MAX_LENGTH,
    BLIP_CACHE_SIZE,
)


class BLIPEncoder:
    """
    Wraps Salesforce BLIP-1 to produce 768-dim semantic embeddings.
    Compatible with transformers==4.30.2 and Python 3.7.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[BLIPEncoder] Loading {} on {} ...".format(
            BLIP_MODEL_NAME, self.device))

        self.processor = BlipProcessor.from_pretrained(
            BLIP_MODEL_NAME, use_fast=True
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME
        ).to(self.device)
        self.model.eval()

        # Verify the correct attribute exists
        assert hasattr(self.model, 'text_decoder'), (
            "Expected model.text_decoder — wrong transformers version?")
        assert hasattr(self.model.text_decoder, 'bert'), (
            "Expected model.text_decoder.bert — wrong BLIP version?")

        print("[BLIPEncoder] Ready. Using text_decoder.bert for embeddings.")

        # LRU-style cache {hash: np.ndarray(768,)}
        self._cache = {}
        self._last_embedding = np.zeros(BLIP_EMBEDDING_DIM, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_embedding(self, image_rgb):
        """
        Parameters
        ----------
        image_rgb : np.ndarray  shape (H, W, 3)  uint8

        Returns
        -------
        embedding : np.ndarray  shape (768,)  float32
        """
        cache_key = hash(image_rgb.tobytes())
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            embedding = self._encode(image_rgb)
        except Exception as exc:
            print("[BLIPEncoder] inference error: {}".format(exc))
            embedding = self._last_embedding.copy()

        # Evict oldest when cache full
        if len(self._cache) >= BLIP_CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))

        self._cache[cache_key] = embedding
        self._last_embedding   = embedding
        return embedding

    def clear_cache(self):
        self._cache.clear()

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    def _encode(self, image_rgb):
        pil_image = Image.fromarray(
            image_rgb.astype(np.uint8)).convert("RGB")

        # ── Step 1 & 2: generate caption ──────────────────────────────
        inputs = self.processor(
            images=pil_image,
            text="Describe the road condition and its surroundings:",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=BLIP_MAX_LENGTH,
                min_length=10,
                num_beams=3,
                no_repeat_ngram_size=2,
            )
        caption = self.processor.decode(
            output_ids[0], skip_special_tokens=True)

        # ── Step 3: tokenize caption ───────────────────────────────────
        text_inputs = self.processor.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=BLIP_MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        # ── Step 4: get hidden states via text_decoder.bert ───────────
        # In transformers 4.30.2 BLIP-1, the BERT backbone sits at
        # model.text_decoder.bert  (NOT model.text_encoder)
        with torch.no_grad():
            bert_outputs = self.model.text_decoder.bert(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                return_dict=True,
            )

        # ── Step 5: mean pool over valid tokens -> (768,) ─────────────
        hidden    = bert_outputs.last_hidden_state           # (1, seq, 768)
        mask      = text_inputs["attention_mask"].unsqueeze(-1).float()
        embedding = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)

        # L2 normalise
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding