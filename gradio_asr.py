# 接口7998

# 带热词的gradio界面

# 添加下载srt/txt文件按钮

import gradio as gr
from funasr import AutoModel
import time
import json

# 初始化模型
model = AutoModel(
    model="damo/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    model_revision="v2.0.4", 
    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_model_revision="v2.0.4",
    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    punc_model_revision="v2.0.4",
    spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
    spk_model_revision="v2.0.2"
)

def time_convert(ms):
    """将毫秒转换为SRT字幕格式的时间字符串"""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt(data):
    srt_content = ""
    for index, item in enumerate(data):
        start_time = time_convert(int(item['timestamp'][0][0] ))  # 转换为毫秒
        end_time = time_convert(int(item['end'] ))  # 转换为毫秒
        speaker = f"speaker{item['spk']}"
        text = item['text']
        srt_content += f"{index + 1}\n{start_time} --> {end_time}\n{speaker}: {text}\n\n"
    return srt_content


def save_to_file(content, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename


def recognize_audio(audio_file, hotwords):
    # 处理音频文件路径
    audio_path = audio_file  
    print(f"音频文件路径: {audio_path}")
    
    # 开始识别
    start_time = time.time()
    res = model.generate(input=audio_path, batch_size_s=300, hotword=hotwords)
    end_time = time.time()

    # 解析返回数据
    data = res[0]["sentence_info"]
    results = []
    for item in data:
        start_sec = item['timestamp'][0][0]
        end_sec = item['end']
        results.append(f"start:{start_sec} end:{end_sec} speaker{item['spk']}:{item['text']}")

    # 生成SRT文件内容
    srt_content = generate_srt(data)

    # 将识别结果保存为txt文件
    txt_filename = save_to_file("\n".join(results), "result.txt")

    # 将识别结果保存为srt文件
    srt_filename = save_to_file(srt_content, "result.srt")

    # 返回识别结果、处理时间、txt文件路径和srt文件路径
    process_time = end_time - start_time
    return "\n".join(results), f"处理时间: {process_time:.2f}秒", txt_filename, srt_filename


with gr.Blocks() as app:
    # 中间标题
    gr.Markdown("<h1 style='text-align: center;'>语音识别——JYD</h1>")
    gr.Markdown("<h1 style='text-align: center;'></h1>")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath",label="上传音频文件")
            hotwords_input = gr.Textbox(label="输入热词")
            submit_button = gr.Button("识别")
        with gr.Column():
            download_txt = gr.File(label="下载识别结果 (TXT)")
            download_srt = gr.File(label="下载字幕文件 (SRT)")
            output_time = gr.Textbox(label="处理时间")
            output_text = gr.Textbox(label="识别结果")
            
    submit_button.click(
        recognize_audio,
        inputs=[audio_input, hotwords_input],
        outputs=[output_text, output_time, download_txt, download_srt]
    )
if __name__ == "__main__":
    app.launch(server_port=7998, server_name='0.0.0.0')
