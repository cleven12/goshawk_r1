"""Binary model exporter and metadata serializer for edge runtime engines."""

import json
import struct
from typing import Any, Dict

from ..config import GoshawkConfig
from ..model.transformer import GoshawkForCausalLM

MAGIC_HEADER = b"GSHK"
FORMAT_VERSION = 1


class GoshawkBinaryExporter:
    """
    Serializes Goshawk R1 weights, vocabulary, and hyperparameters into a compact binary format.
    Allows zero-dependency execution on embedded Linux, CCTV NVRs, and mobile runtimes.
    """

    @staticmethod
    def export(
        model: GoshawkForCausalLM,
        config: GoshawkConfig,
        output_path: str,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Serialize model into binary file."""
        meta = metadata or {}
        meta_dict = {
            "arch": "goshawk_r1",
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "v_head_dim": config.v_head_dim,
            "vocab_size": config.vocab_size,
            "q_lora_rank": config.q_lora_rank,
            "kv_lora_rank": config.kv_lora_rank,
            "n_routed_experts": config.n_routed_experts,
            "n_shared_experts": config.n_shared_experts,
            "extra": meta,
        }
        meta_bytes = json.dumps(meta_dict).encode("utf-8")

        state_dict = model.state_dict()
        tensor_names = sorted(state_dict.keys())

        # Construct file layout:
        # [Header: 4 bytes Magic]
        # [Version: 4 bytes uint32]
        # [Metadata length: 4 bytes uint32]
        # [Metadata JSON bytes]
        # [Tensor count: 4 bytes uint32]
        # For each tensor:
        #   [Name length: 2 bytes uint16]
        #   [Name UTF-8]
        #   [Num dimensions: 2 bytes uint16]
        #   [Dimensions: N * 4 bytes uint32]
        #   [Data length: 8 bytes uint64]
        #   [Raw Float32 Data]

        total_bytes = 0
        with open(output_path, "wb") as f:
            f.write(MAGIC_HEADER)
            f.write(struct.pack("<I", FORMAT_VERSION))
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            f.write(struct.pack("<I", len(tensor_names)))

            for name in tensor_names:
                tensor = state_dict[name].cpu().float().contiguous()
                name_bytes = name.encode("utf-8")
                shape = list(tensor.shape)
                data_bytes = tensor.numpy().tobytes()

                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<H", len(shape)))
                for dim in shape:
                    f.write(struct.pack("<I", dim))
                f.write(struct.pack("<Q", len(data_bytes)))
                f.write(data_bytes)
                total_bytes += len(data_bytes)

        return {
            "output_path": output_path,
            "tensors_exported": len(tensor_names),
            "payload_bytes": total_bytes,
            "metadata": meta_dict,
        }

    @staticmethod
    def inspect(file_path: str) -> Dict[str, Any]:
        """Read and inspect header metadata from binary export."""
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC_HEADER:
                raise ValueError(f"Invalid file format: expected {MAGIC_HEADER}, got {magic}")
            (version,) = struct.unpack("<I", f.read(4))
            (meta_len,) = struct.unpack("<I", f.read(4))
            meta_json = f.read(meta_len).decode("utf-8")
            (num_tensors,) = struct.unpack("<I", f.read(4))

        return {
            "version": version,
            "metadata": json.loads(meta_json),
            "num_tensors": num_tensors,
        }
