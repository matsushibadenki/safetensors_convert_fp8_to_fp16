# safetensors convert fp8 to fp16
Comfyul is specialized for poor fp8 users, so it needs to be converted to fp16 in the MacOS environment.
This script converts fp8 safetensors files to fp16.


```bash
pip install torch safetensors tqdm
```

```bash
python convert_fp8_to_fp16.py [変換元のファイルパス] [変換後のファイルパス]
```
