# 接口的降噪模块，千万不能删

# 修改模型加载为绝对路径

import torch
import tempfile
import numpy as np
from loguru import logger
from torch import Tensor
import gc  # 导入垃圾回收模块
import os

from df import config
from df.enhance import enhance, init_df, load_audio, save_audio
from df.io import resample




# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 模型路径改为绝对路径
model_base_dir = os.path.join(current_dir, "DeepFilterNet2")



# 初始化模型和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, df, _ = init_df(model_base_dir, config_allow_defaults=True)      # 绝对路径初始化方法
# model, df, _ = init_df("./DeepFilterNet2", config_allow_defaults=True)    # 相对路径初始化方法

model = model.to(device=device).eval()

def denoise_audio(input_audio_path: str, output_audio_path: str = None) -> str:
    try:
        # 设置采样率
        sr = 48000

        # 加载输入音频
        print("开始加载音频")
        sample, meta = load_audio(input_audio_path, sr)
        if sample.dim() > 1 and sample.shape[0] > 1:
            sample = sample.mean(dim=0, keepdim=True)

        # 降噪处理
        enhanced = enhance(model, df, sample)

        # 重采样以匹配原始采样率
        if meta.sample_rate != sr:
            enhanced = resample(enhanced, sr, meta.sample_rate)

        # 保存降噪后的音频
        if output_audio_path is None:
            output_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        save_audio(output_audio_path, enhanced, meta.sample_rate)

        # 清理显存和内存
        del sample, enhanced  # 删除不再需要的变量
        torch.cuda.empty_cache()  # 释放显存
        gc.collect()  # 触发垃圾回收

        return output_audio_path

    except Exception as e:
        logger.error(f"降噪过程中出错: {e}")

        # 确保异常时也释放显存和内存
        torch.cuda.empty_cache()
        gc.collect()

        raise e  # 重新抛出异常，以便上层捕获
