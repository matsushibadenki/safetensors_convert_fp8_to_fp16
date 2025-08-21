# path: ./convert_fp8_to_fp16.py
# title: Safetensors FP8 to FP16 Converter
# role: Converts all FP8 tensors within a .safetensors file to FP16 format, which is more compatible with Mac environments.

import argparse
import os
from typing import Dict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def convert_fp8_to_fp16(source_path: str, target_path: str) -> None:
    """
    Loads a safetensors file, converts all FP8 tensors to Float16,
    and saves the result to a new safetensors file.

    Args:
        source_path (str): The path to the source .safetensors file.
        target_path (str): The path to save the converted .safetensors file.
    """
    if not os.path.exists(source_path):
        print(f"エラー: 指定されたソースファイルが見つかりません: '{source_path}'")
        return

    print(f"ファイルをロード中: '{source_path}'...")
    # ファイル全体をCPUメモリにロードします。
    # 巨大なファイルの場合、多くのメモリが必要になる点に注意してください。
    tensors: Dict[str, torch.Tensor] = load_file(source_path, device="cpu")

    converted_tensors: Dict[str, torch.Tensor] = {}

    print("テンソルをFP8からFP16へ変換中...")
    # tqdmを使って進捗バーを表示します
    for name, tensor in tqdm(tensors.items(), desc="テンソルを処理中"):
        # テンソルのデータ型がFP8の亜種（torch.float8_e4m3fnなど）かを確認します。
        if 'float8' in str(tensor.dtype):
            # テンソルをFloat16（半精度）に変換します。
            converted_tensors[name] = tensor.to(torch.float16)
        else:
            # FP8でない場合は、元のデータ型のまま保持します。
            converted_tensors[name] = tensor

    print(f"変換後のファイルを保存中: '{target_path}'...")
    # 変換されたテンソルを新しいファイルに保存します。
    save_file(converted_tensors, target_path)

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