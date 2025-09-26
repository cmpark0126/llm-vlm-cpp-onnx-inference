import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM


class TempCache:
    def __init__(self):
        pass

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        self.key_states = key_states
        self.value_states = value_states
        self.layer_idx = layer_idx
        self.cache_kwargs = cache_kwargs

        return self.key_states, self.value_states


class StaticGemmaPrefill(nn.Module):
    """
    Gemma3 Prefill stage with static shapes for ONNX export
    Fixed sequence length: 128
    """

    def __init__(self, original_model, config):
        super().__init__()
        self.config = config
        self.seq_length = 128

        # Copy layers from original model
        self.embed_tokens = original_model.model.embed_tokens
        self.layers = original_model.model.layers[: config.num_hidden_layers]
        self.norm = original_model.model.norm
        self.rotary_emb = original_model.model.rotary_emb
        self.rotary_emb_local = original_model.model.rotary_emb_local
        self.lm_head = original_model.lm_head

        # Pre-compute static masks and positions
        self._prepare_static_components()

    def _prepare_static_components(self):
        """Pre-compute static components that don't change during inference"""

        # 1. Static cache position (0 to 127 for prefill)
        self.register_buffer(
            "static_cache_position", torch.arange(0, self.seq_length, dtype=torch.long)
        )

        # 2. Static position IDs (same as cache_position for prefill)
        self.register_buffer(
            "static_position_ids",
            torch.arange(0, self.seq_length, dtype=torch.long).unsqueeze(0),  # [1, 128]
        )

        # 3. Pre-compute causal masks for both attention types
        self._prepare_static_masks()

        # 4. Initialize static KV cache structure
        self._prepare_static_kv_cache()

    def _prepare_static_masks(self):
        """Pre-compute static attention masks"""

        # Create basic causal mask (lower triangular)
        causal_mask = torch.triu(
            torch.full((self.seq_length, self.seq_length), float("-inf")), diagonal=1
        )

        # Full attention mask (standard causal mask)
        self.register_buffer("static_full_attention_mask", causal_mask)

        # Sliding window mask (if applicable)
        if (
            hasattr(self.config, "sliding_window")
            and self.config.sliding_window is not None
        ):
            sliding_window = self.config.sliding_window
            sliding_mask = causal_mask.clone()

            # Apply sliding window - mask tokens beyond window size
            for i in range(self.seq_length):
                start_pos = max(0, i - sliding_window)
                if start_pos > 0:
                    sliding_mask[i, :start_pos] = float("-inf")

            self.register_buffer("static_sliding_attention_mask", sliding_mask)
        else:
            self.register_buffer("static_sliding_attention_mask", causal_mask)

    def _prepare_static_kv_cache(self):
        """Initialize static KV cache tensors"""

        num_layers = len(self.layers)
        num_key_value_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        # Initialize empty KV cache with fixed shape
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        self.static_k_cache_shape = (1, num_key_value_heads, self.seq_length, head_dim)
        self.static_v_cache_shape = (1, num_key_value_heads, self.seq_length, head_dim)

        # Create cache containers for each layer
        self.register_buffer("cache_layers_info", torch.tensor(num_layers))

    def _create_static_attention_mask(
        self, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert input attention mask to proper format for static computation
        attention_mask: [batch_size, seq_length] - 1 for tokens, 0 for padding
        """
        batch_size = attention_mask.shape[0]

        # Expand attention mask to [batch_size, 1, seq_length, seq_length]
        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        expanded_mask = expanded_mask.expand(
            batch_size, 1, self.seq_length, self.seq_length
        )

        # Convert to additive mask (0 for attend, -inf for ignore)
        inverted_mask = 1.0 - expanded_mask
        attention_mask_float = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), float("-inf")
        )

        return attention_mask_float

    def forward(
        self,
        input_ids: torch.LongTensor,  # [batch_size, 128]
        attention_mask: torch.Tensor,  # [batch_size, 128]
        position_ids: Optional[torch.LongTensor] = None,  # [batch_size, 128] or None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Static forward pass for prefill stage

        Returns:
            Tuple containing:
            - last_hidden_state: [batch_size, 128, hidden_size]
            - present.0.key: [batch_size, num_key_value_heads, 128, head_dim]
            - present.0.value: [batch_size, num_key_value_heads, 128, head_dim]
            - present.1.key: [batch_size, num_key_value_heads, 128, head_dim]
            - present.1.value: [batch_size, num_key_value_heads, 128, head_dim]
            - ... (continues for all layers)
            - present.{num_layers-1}.key: [batch_size, num_key_value_heads, 128, head_dim]
            - present.{num_layers-1}.value: [batch_size, num_key_value_heads, 128, head_dim]
        """

        batch_size = input_ids.shape[0]

        # 1. Input validation - ensure static shape
        assert (
            input_ids.shape[1] == self.seq_length
        ), f"Expected seq_len {self.seq_length}, got {input_ids.shape[1]}"
        assert (
            attention_mask.shape[1] == self.seq_length
        ), f"Expected seq_len {self.seq_length}, got {attention_mask.shape[1]}"

        # 2. Embed tokens
        inputs_embeds = self.embed_tokens(input_ids)  # [batch_size, 128, hidden_size]

        # 3. Use static position_ids if not provided
        if position_ids is None:
            position_ids = self.static_position_ids.expand(batch_size, -1)

        # 4. Create attention masks for both attention types
        input_attention_mask = self._create_static_attention_mask(attention_mask)

        # Combine with causal masks
        full_attention_mask = (
            input_attention_mask + self.static_full_attention_mask.unsqueeze(0)
        )
        sliding_attention_mask = (
            input_attention_mask + self.static_sliding_attention_mask.unsqueeze(0)
        )

        causal_mask_mapping = {
            "full_attention": full_attention_mask,
            "sliding_attention": sliding_attention_mask,
        }

        # 5. Create position embeddings
        hidden_states = inputs_embeds

        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # 6. Initialize KV caches for each layer
        all_kv_caches = []

        # 7. Process through decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):

            # Initialize empty KV cache for this layer
            device = hidden_states.device
            dtype = hidden_states.dtype

            kv_cache = TempCache()

            # Get appropriate attention mask for this layer
            attention_mask_for_layer = causal_mask_mapping[decoder_layer.attention_type]

            # Forward through layer
            outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=attention_mask_for_layer,
                position_ids=position_ids,
                past_key_values=kv_cache,
                output_attentions=False,
                use_cache=True,
                cache_position=self.static_cache_position.expand(batch_size, -1),
            )

            hidden_states = outputs[0]

            # Extract updated KV cache
            all_kv_caches.append((kv_cache.key_states, kv_cache.value_states))

        # 8. Final layer norm
        hidden_states = self.norm(hidden_states)

        # 9. Generate logits
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        # 10. Flatten KV caches to individual tensors for ONNX export
        # Format: logits, present.0.key, present.0.value, present.1.key, present.1.value, ...
        flattened_outputs = [logits]

        for layer_idx, (k_cache, v_cache) in enumerate(all_kv_caches):
            flattened_outputs.extend([k_cache, v_cache])

        return tuple(flattened_outputs)


def export_static_gemma_prefill_to_onnx(
    original_model, config, output_path: str, batch_size: int = 1
):
    """
    Export StaticGemmaPrefill to ONNX format
    """

    # Create static model
    static_model = StaticGemmaPrefill(original_model, config)
    static_model.eval()

    # Create dummy inputs with static shapes
    dummy_input_ids = torch.randint(
        0, config.vocab_size, (batch_size, 128), dtype=torch.long
    )
    dummy_attention_mask = torch.ones((batch_size, 128), dtype=torch.long)
    # Position IDs는 항상 [1, 2, 3, ..., 128]로 고정
    dummy_position_ids = (
        torch.arange(1, 129, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )

    # Create output names for individual KV cache tensors
    # Format: logits, present.0.key, present.0.value, present.1.key, present.1.value, ...
    output_names = ["logits"]

    num_layers = config.num_hidden_layers
    for layer_idx in range(num_layers):
        output_names.extend([f"present.{layer_idx}.key", f"present.{layer_idx}.value"])

    print(f"Exporting with {len(output_names)} outputs:")
    print(f"  - logits")
    for layer_idx in range(num_layers):
        print(f"  - present.{layer_idx}.key, present.{layer_idx}.value")

    # Export to ONNX
    torch.onnx.export(
        static_model,
        (dummy_input_ids, dummy_attention_mask, dummy_position_ids),
        output_path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=output_names,
        dynamic_axes=None,  # No dynamic axes - all static
        opset_version=17,
        do_constant_folding=True,
        verbose=True,
    )

    print(f"Static Gemma Prefill exported to {output_path}")
    print(
        f"Total outputs: {len(output_names)} (1 hidden_state + {num_layers * 2} KV caches)"
    )


if __name__ == "__main__":
    print("Loading model...")
    local_model_path = "./gemma-3-1b-it"
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        cache_dir=local_model_path,  # 현재 폴더에 다운로드
    )

    print("Creating static model...")
    static_model = StaticGemmaPrefill(model, model.config)

    print("Exporting to ONNX...")
    export_static_gemma_prefill_to_onnx(
        model, model.config, "gemma-3-1b-it-prefill.onnx"
    )
