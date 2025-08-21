# path: ./convert_fp8_to_fp16.py
# title: Safetensors FP8 to FP16 Converter with Progress Bar
# role: Converts all FP8 tensors within a .safetensors file to FP16 format, 
#       and displays progress for both conversion and saving processes.

import argparse
import json
import os
from typing import Dict, Optional

import torch
from safetensors.torch import load_file
from tqdm import tqdm


def save_file_with_progress(
    tensors: Dict[str, torch.Tensor],
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Saves tensors to a safetensors file with a progress bar for the I/O operation.
    This function mimics the internal logic of safetensors.torch.save_file
    but adds tqdm for progress visualization.
    """
    if metadata is None:
        metadata = {}

    # 1. Prepare header metadata
    headers = {"__metadata__": metadata}
    for name, tensor in tensors.items():
        headers[name] = {
            # PyTorch dtype like 'torch.float16' is serialized to just 'F16'
            "dtype": str(tensor.dtype).split(".")[-1],
            "shape": list(tensor.shape),
        }

    # 2. Serialize header to a JSON string
    header_str = json.dumps(headers, separators=(",", ":"), ensure_ascii=False)
    header_bytes = header_str.encode("utf-8")
    
    # Prepend header size (8-byte unsigned little-endian integer)
    header_size_info = len(header_bytes).to_bytes(8, "little")

    # 3. Calculate total file size for the progress bar
    total_size = len(header_size_info) + len(header_bytes)
    for tensor in tensors.values():
        total_size += tensor.numel() * tensor.element_size()

    print(f"変換後のファイルを保存中 (合計サイズ: {total_size / (1024*1024):.2f} MB): '{filename}'...")

    # 4. Write to file with progress bar
    with open(filename, "wb") as f, tqdm(
        total=total_size,
        desc="ファイルを保存中",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        # Write header size and header content
        f.write(header_size_info)
        f.write(header_bytes)
        pbar.update(len(header_size_info) + len(header_bytes))

        # Write each tensor's data sequentially
        for tensor in tensors.values():
            # Ensure tensor is on CPU and in a contiguous memory layout before saving
            tensor_bytes = tensor.contiguous().cpu().numpy().tobytes()
            f.write(tensor_bytes)
            pbar.update(len(tensor_bytes))


def convert_fp8_to_fp16(source_path: str, target_path: str) -> None:
    """
    Loads a safetensors file, converts all FP8 tensors to Float16,
    and saves the result to a new safetensors file.
    """
    if not os.path.exists(source_path):
        print(f"エラー: 指定されたソースファイルが見つかりません: '{source_path}'")
        return

    print(f"ファイルをロード中: '{source_path}'...")
    tensors: Dict[str, torch.Tensor] = load_file(source_path, device="cpu")

    converted_tensors: Dict[str, torch.Tensor] = {}

    print("テンソルをFP8からFP16へ変換中...")
    for name, tensor in tqdm(tensors.items(), desc="テンソルを処理中"):
        if 'float8' in str(tensor.dtype):
            converted_tensors[name] = tensor.to(torch.float16)
        else:
            converted_tensors[name] = tensor
    
    # Use the new function with progress bar for saving
    save_file_with_progress(converted_tensors, target_path)

    print("\n変換が完了しました！ ✨")
    try:
        source_size_mb = os.path.getsize(source_path) / (1024 * 1024)
        target_size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"元のファイルサイズ: {source_size_mb:.2f} MB")
        print(f"変換後のファイルサイズ: {target_size_mb:.2f} MB")
    except FileNotFoundError:
        print("ファイルサイズの取得中にエラーが発生しました。")


def main() -> None:
    """
    コマンドライン引数を解析し、変換処理を実行するメイン関数。
    """
    parser = argparse.ArgumentParser(
        description="safetensorsファイル内のFP8テンソルをFP16に変換します。"
    )
    parser.add_argument(
        "source_file",
        type=str,
        help="変換元のsafetensorsファイルへのパス (例: model.fp8.safetensors)"
    )
    parser.add_argument(
        "target_file",
        type=str,
        help="変換後のFP16 safetensorsファイルを保存するパス (例: model.fp16.safetensors)"
    )

    args = parser.parse_args()
    convert_fp8_to_fp16(args.source_file, args.target_file)


if __name__ == "__main__":
    main()