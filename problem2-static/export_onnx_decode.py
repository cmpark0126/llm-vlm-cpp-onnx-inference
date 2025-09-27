import os

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from transformers import AutoModelForCausalLM


@torch.jit.script
def update_cache(existing_key_cache: torch.Tensor, existing_value_cache: torch.Tensor,
                 new_key_states: torch.Tensor, new_value_states: torch.Tensor,
                 current_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    새로운 key/value를 기존 cache의 다음 위치에 추가
    """
    updated_key_cache = existing_key_cache.clone()
    updated_value_cache = existing_value_cache.clone()

    # current_length 위치에 새로운 key/value 추가 - scatter 사용
    batch_size, num_heads, _, head_dim = updated_key_cache.shape

    # 인덱스 텐서 생성: current_length를 모든 차원에 확장
    pos_indices = current_length.view(1, 1, 1, 1).expand(batch_size, num_heads, 1, head_dim)

    # scatter를 사용하여 업데이트
    updated_key_cache.scatter_(2, pos_indices, new_key_states)
    updated_value_cache.scatter_(2, pos_indices, new_value_states)

    return updated_key_cache, updated_value_cache


class TempCache:
    def __init__(
        self, existing_key_cache=None, existing_value_cache=None, current_length=None
    ):
        """
        Args:
            existing_key_cache: [batch_size, num_heads, max_cache_length, head_dim]
            existing_value_cache: [batch_size, num_heads, max_cache_length, head_dim]
            current_length: 현재 유효한 sequence length (tensor)
        """
        self.key_states = existing_key_cache
        self.value_states = existing_value_cache
        self.current_length = current_length

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: Any, cache_kwargs: Any):
        """
        새로운 key/value를 기존 cache의 다음 위치에 추가
        key_states, value_states: [batch_size, num_heads, 1, head_dim] - 새로운 토큰
        """
        # TorchScript 함수 사용
        self.key_states, self.value_states = update_cache(
            self.key_states, self.value_states, key_states, value_states, self.current_length
        )

        return self.key_states, self.value_states

class StaticGemmaDecode(nn.Module):
    """
    Gemma3 Decode stage with static shapes for ONNX export
    Fixed cache length: 1024, input sequence length: 1
    """

    def __init__(self, original_model, config):
        super().__init__()
        self.config = config
        self.cache_length = 1024
        self.input_seq_length = 1

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

        # 1. Static cache position (0 to 1023 for cache)
        self.register_buffer(
            "static_cache_position", torch.arange(0, self.cache_length, dtype=torch.long)
        )

        # 2. Static position IDs for input token (will be dynamic based on current position)
        # This is just a placeholder - actual position_ids will be passed as input
        self.register_buffer(
            "static_position_ids",
            torch.zeros(1, self.input_seq_length, dtype=torch.long),  # [1, 1]
        )

        # 3. Pre-compute causal masks for both attention types
        self._prepare_static_masks()

        # 4. Initialize static KV cache structure
        self._prepare_static_kv_cache()

    def _prepare_static_masks(self):
        """Pre-compute static attention masks"""

        # For decode stage, we don't need pre-computed masks since attention
        # is computed dynamically based on current position and cache length
        # Masks will be created dynamically in forward pass
        pass

    def _prepare_static_kv_cache(self):
        """Initialize static KV cache tensors"""

        num_layers = len(self.layers)
        num_key_value_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        head_dim = self.config.head_dim

        # Initialize KV cache with fixed shape for 1024 positions
        # Shape: [batch_size, num_heads, cache_length, head_dim]
        self.static_k_cache_shape = (1, num_key_value_heads, self.cache_length, head_dim)
        print(f"static_k_cache_shape: {self.static_k_cache_shape}")
        self.static_v_cache_shape = (1, num_key_value_heads, self.cache_length, head_dim)
        print(f"static_v_cache_shape: {self.static_v_cache_shape}")

        # Create cache containers for each layer
        self.register_buffer("cache_layers_info", torch.tensor(num_layers))


    def forward(
        self,
        input_ids: torch.LongTensor,  # [batch_size, 1]
        position_ids: torch.LongTensor,  # [batch_size, 1]
        attention_mask: torch.Tensor,  # [batch_size, 1024]
        cache_position: torch.Tensor,  # Target position in cache for new key/value [1]
        *past_key_values,  # Previous KV cache for each layer [batch_size, num_heads, 1024, head_dim]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Static forward pass for decode stage

        Args:
            input_ids: [batch_size, 1] - Single token input
            position_ids: [batch_size, 1] - Current position in sequence
            attention_mask: [batch_size, 1024] - Attention mask (1 for valid positions, 0 for padding)
            cache_position: torch.Tensor - Target position in cache for new key/value (0-based) [1]
            *past_key_values: Previous KV cache for each layer,
                             each pair: (key, value) with shape [batch_size, num_heads, 1024, head_dim]

        Returns:
            Tuple containing:
            - logits: [batch_size, 1, vocab_size]
            - present.0.key: [batch_size, num_key_value_heads, 1024, head_dim]
            - present.0.value: [batch_size, num_key_value_heads, 1024, head_dim]
            - ... (continues for all layers)
        """

        batch_size = input_ids.shape[0]

        # 1. Input validation - ensure single token input
        assert (
            input_ids.shape[1] == self.input_seq_length
        ), f"Expected seq_len {self.input_seq_length}, got {input_ids.shape[1]}"
        assert (
            position_ids.shape[1] == self.input_seq_length
        ), f"Expected seq_len {self.input_seq_length}, got {position_ids.shape[1]}"

        # 2. Parse past KV caches
        num_layers = len(self.layers)
        assert len(past_key_values) == num_layers * 2, f"Expected {num_layers * 2} past KV tensors, got {len(past_key_values)}"

        past_kv_pairs = []
        for i in range(0, len(past_key_values), 2):
            past_key = past_key_values[i]    # [batch_size, num_heads, 1024, head_dim]
            past_value = past_key_values[i+1]  # [batch_size, num_heads, 1024, head_dim]
            past_kv_pairs.append((past_key, past_value))

        # 3. Embed tokens
        inputs_embeds = self.embed_tokens(input_ids)  # [batch_size, 1, hidden_size]

        # 4. Create position embeddings
        hidden_states = inputs_embeds
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # 5. Process through decoder layers
        all_kv_caches = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            # Get past KV cache for this layer
            past_key, past_value = past_kv_pairs[layer_idx]

            # Create TempCache with existing cache and current position
            kv_cache = TempCache(
                existing_key_cache=past_key,
                existing_value_cache=past_value,
                current_length=cache_position
            )

            # Use the provided attention mask
            # Convert from [batch_size, 1024] to [batch_size, 1, 1, 1024] format
            layer_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert to additive mask: 0 for attend, -inf for ignore
            inverted_mask = 1.0 - layer_attention_mask
            layer_attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), float("-inf")
            )

            # Forward through layer
            outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_values=kv_cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]

            # Extract updated KV cache (should be same as past cache with new token added)
            all_kv_caches.append((kv_cache.key_states, kv_cache.value_states))

        # 6. Final layer norm
        hidden_states = self.norm(hidden_states)

        # 7. Generate logits
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        # 8. Flatten KV caches to individual tensors for ONNX export
        # Format: logits, present.0.key, present.0.value, present.1.key, present.1.value, ...
        flattened_outputs = [logits]

        for layer_idx, (k_cache, v_cache) in enumerate(all_kv_caches):
            flattened_outputs.extend([k_cache, v_cache])

        return tuple(flattened_outputs)


def export_static_gemma_decode_to_onnx(
    original_model, config, output_path: str, batch_size: int = 1
):
    """
    Export StaticGemmaDecode to ONNX format
    """

    # Create static model
    static_model = StaticGemmaDecode(original_model, config)
    static_model.eval()

    num_layers = config.num_hidden_layers
    num_key_value_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads
    )
    head_dim = config.head_dim
    print(f"num_layers: {num_layers}, num_key_value_heads: {num_key_value_heads}, head_dim: {head_dim}")

    # Create dummy inputs with static shapes
    dummy_input_ids = torch.randint(
        0, config.vocab_size, (batch_size, 1), dtype=torch.long
    )
    dummy_position_ids = torch.tensor([[0]], dtype=torch.long)  # Start at position 0
    dummy_cache_position = torch.tensor([0], dtype=torch.long)  # Target cache position
    dummy_attention_mask = torch.ones((batch_size, 1024), dtype=torch.long)  # All positions valid for testing

    # Create dummy past KV caches (initialized with zeros)
    dummy_past_kv = []
    for layer_idx in range(num_layers):
        dummy_key = torch.zeros((batch_size, num_key_value_heads, 1024, head_dim), dtype=torch.float32)
        dummy_value = torch.zeros((batch_size, num_key_value_heads, 1024, head_dim), dtype=torch.float32)
        dummy_past_kv.extend([dummy_key, dummy_value])

    # Create input names for KV caches
    input_names = ["input_ids", "position_ids", "attention_mask", "cache_position"]
    for layer_idx in range(num_layers):
        input_names.extend([f"past.{layer_idx}.key", f"past.{layer_idx}.value"])

    # Create output names for individual KV cache tensors
    # Format: logits, present.0.key, present.0.value, present.1.key, present.1.value, ...
    output_names = ["logits"]
    for layer_idx in range(num_layers):
        output_names.extend([f"present.{layer_idx}.key", f"present.{layer_idx}.value"])

    print(f"Exporting with {len(input_names)} inputs:")
    print(f"  - input_ids: [1, 1]")
    print(f"  - position_ids: [1, 1]")
    print(f"  - attention_mask: [1, 1024]")
    print(f"  - cache_position: [1]")
    for layer_idx in range(num_layers):
        print(f"  - past.{layer_idx}.key, past.{layer_idx}.value: [1, {num_key_value_heads}, 1024, {head_dim}]")

    print(f"Exporting with {len(output_names)} outputs:")
    print(f"  - logits: [1, 1, {config.vocab_size}]")
    for layer_idx in range(num_layers):
        print(f"  - present.{layer_idx}.key, present.{layer_idx}.value: [1, {num_key_value_heads}, 1024, {head_dim}]")

    # Prepare arguments for export
    export_args = (dummy_input_ids, dummy_position_ids, dummy_attention_mask, dummy_cache_position) + tuple(dummy_past_kv)

    # Export to ONNX
    torch.onnx.export(
        static_model,
        export_args,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,  # No dynamic axes - all static
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"Static Gemma Decode exported to {output_path}")
    print(
        f"Total inputs: {len(input_names)} (3 main inputs + {num_layers * 2} past KV caches)"
    )
    print(
        f"Total outputs: {len(output_names)} (1 logits + {num_layers * 2} present KV caches)"
    )


if __name__ == "__main__":
    print("Loading model...")
    local_model_path = "./gemma-3-1b-it"
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        cache_dir=local_model_path,  # 현재 폴더에 다운로드
    )

    print("Creating static model...")
    static_model = StaticGemmaDecode(model, model.config)

    print("Exporting to ONNX...")
    if not os.path.exists("gemma-3-1b-it-decode"):
        os.makedirs("gemma-3-1b-it-decode")

    print("Exporting to ONNX...")
    export_static_gemma_decode_to_onnx(
        model, model.config, "gemma-3-1b-it-decode/gemma-3-1b-it-decode.onnx"
    )
