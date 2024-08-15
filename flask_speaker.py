# 接口：8006

# paraformer + cam++
# 带降噪 DeepFilterNet2降噪（切分音频）
# 需要前期将音频转换为左声道（目前效果最好）
# 带定量清理文件，删除最早的文件（防止内存占满）
# 分离左声道

# 处理静默时间 小于2s的归到上一条字幕
# 长时间静默后出现声音，整段时间都会标记到字幕行（瑞华）
# 音频统一处理为16000hz

# 字幕合并：以？。！做结尾，合并后计算字幕长度，大于60不继续合并；
# 处理端点检测模型失误情况 
# 再次处理静默时间 小于2s的归到上一条字幕



from flask import Flask, request, jsonify, url_for,send_from_directory
import os
import re
import csv
import time
import argparse
import pandas as pd
from funasr import AutoModel
import threading
import torch
from pydub import AudioSegment
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from datetime import datetime

app = Flask(__name__)

# 上传的文件和结果文件保存在这个目录下
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# 跟踪音频处理状态的字典
audio_processing_status = {}



def manage_files(directory, max_files=9):
    """管理文件夹中的文件数量，保持文件数量不超过 max_files，如果超过就删除最旧的文件"""
    files = os.listdir(directory)
    if len(files) > max_files:
        # 获取每个文件的完整路径和修改时间
        full_paths = [os.path.join(directory, file) for file in files]
        # 按照修改时间排序文件
        full_paths = sorted(full_paths, key=lambda x: os.path.getmtime(x))
        # 删除最旧的 delete_count 个文件
        for file in full_paths[:len(files)-max_files]:
            os.remove(file)
            print(f"删除 {file} 由于文件数量超过限制")

def save_and_manage_file(file, path):
    """保存文件并管理文件夹中的文件数量"""
    file.save(path)
    # 管理 uploads 文件夹
    manage_files(UPLOAD_FOLDER)
    # 管理 results 文件夹
    manage_files(RESULT_FOLDER)

def convert_audio_to_left_channel(input_file, output_dir):
    """
    将音频文件转换为左声道，并保存到指定目录。
    
    Args:
    input_file (str): 输入音频文件路径。
    output_dir (str): 输出目录路径。
    
    Returns:
    str: 左声道音频文件的路径。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载音频文件
    audio = AudioSegment.from_file(input_file)

    # 将音频采样率转换为16000Hz
    audio = audio.set_frame_rate(16000)
    print("音频采样率降为16000hz")

    # 检查音频是否是立体声
    if audio.channels == 2:
        left_channel = audio.split_to_mono()[0]
        # 获取文件名
        filename, file_extension = os.path.splitext(os.path.basename(input_file))
        # 保存左声道音频文件
        left_channel_path = os.path.join(output_dir, f"{filename}_left{file_extension}")
        # left_channel_path = os.path.join(output_dir, "left_channel.wav")
        left_channel.export(left_channel_path, format="wav")
        print("立体声音频已转换为左声道，保存路径为：", left_channel_path)
        return left_channel_path
    else:
        print("原音频是单声道，直接使用原文件。")
        return input_file
    
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

def time_convert_vtt(ms):
    """将毫秒转换为WebVTT字幕格式的时间字符串"""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def generate_vtt(data):
    vtt_content = "WEBVTT\n\n"
    for index, item in enumerate(data):
        start_time = time_convert_vtt(int(item['timestamp'][0][0]))  # 转换为毫秒
        end_time = time_convert_vtt(int(item['end']))  # 转换为毫秒
        speaker = f"speaker{item['spk']}"
        text = item['text']
        vtt_content += f"{start_time} --> {end_time}\n{speaker}: {text}\n\n"
    return vtt_content

def classify_sentences(speaker_info):
    print("\n**************************句子分类**************************\n")
    with open('./configs/question_words.txt', 'r', encoding='utf-8') as file:
        question_words = [word.strip() for word in file.readlines()]
    
    with open('./configs/noquestion_words.txt', 'r', encoding='utf-8') as file:
        noquestion_words = [word.strip() for word in file.readlines()]

    def is_question(sentence):
        for word in noquestion_words:
            if re.search(re.escape(word), sentence):
                return False

        for word in question_words:
            if re.search(re.escape(word), sentence):
                return True

        return False
    
    sentences = [info.split(':')[1] for info in speaker_info]  # 提取每个speaker的句子
    question_sentences = [sentence for sentence in sentences if is_question(sentence)]
    
    print("\n************************课堂诊断结束************************\n")
    return question_sentences

def process_audio_and_save_result(audio_filepath, audioid):
    
    start_time = time.time()
    print("start time: 开始计时")
    try:
        audioid = audioid

        input_mp3_path = audio_filepath
        temp_folder = 'temp_audio_chunks'  # 用于存储音频片段的临时文件夹
        output_audio_path = f'temp_audio_chunks/processed_audio.wav'  # 指定输出文件的路径

        # 将音频转换为左声道
        left_channel_audio_path = convert_audio_to_left_channel(input_mp3_path, temp_folder)

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
        
        res = model.generate(input=left_channel_audio_path,batch_size_s=300,hotword='基因')
        # 删除left_channel_audio_path
        os.remove(left_channel_audio_path)
    except Exception as e:
        if "CUDA out of memory" in str(e):
            audio_processing_status[audioid] = 'nocuda'  # 更新状态为失败
            # 显示异常信息
            print(f"处理音频文件错误，异常信息为：{e}")
        else:
            audio_processing_status[audioid] = 'failed'  # 更新状态为失败
            print(f"处理音频文件错误，异常信息为: {e}")
    finally:
        # 清理PyTorch的CUDA缓存
        torch.cuda.empty_cache()  # 释放未被引用的内存

    # 检查res列表是否为空
    if not res or "sentence_info" not in res[0]:
        print("没有从音频文件中识别出任何内容")
        audio_processing_status[audioid] = 'failed'  # 更新状态为失败
        return  
    
    # 打印出文本
    data = res[0]["sentence_info"]
 
    # 保存结果到 CSV 文件
    result_filename = f'{audioid}_speakers.csv'
    result_filepath = os.path.join(RESULT_FOLDER, result_filename)


    # 初始化一个变量来存储上一行的 `end` 时间
    previous_end_time = None

    with open(result_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头，增加 'interval' 列
        writer.writerow(['start', 'end', 'interval', 'speaker', 'text'])
        
        for item in data:
            start_sec = item['timestamp'][0][0] / 1000
            end_sec = item['end'] / 1000
            

            # 计算时间间隔，如果这不是第一行的话
            interval = start_sec - previous_end_time if previous_end_time is not None else item['timestamp'][0][0] / 1000

             # 第一行特殊处理
            if previous_end_time is None :
                writer.writerow([0, start_sec,interval,'', ''])
            else:
                # 将数据行写入CSV文件，包括时间间隔
                writer.writerow([previous_end_time, start_sec, interval, '', ''])

            speaker_interval = end_sec-start_sec

            
            writer.writerow([start_sec, end_sec, speaker_interval, item.get('spk', ''), item.get('text', '')])
            

            # 更新上一行的 `end` 时间
            previous_end_time = end_sec

    def process_dataframe1(df):
        i = 0
        while i < len(df) - 2:  # 修改条件以避免索引越界
            # 检查当前行和隔行的speaker是否相等且不为空，且text长度小于等于4
            if pd.notnull(df.loc[i, 'speaker']) and \
            ((pd.notnull(df.loc[i + 2, 'speaker']) and df.loc[i, 'speaker'] == df.loc[i + 2, 'speaker'] and len(str(df.loc[i, 'text'])) <= 6) or \
                str(df.loc[i, 'text']).endswith("、")):
                # 暂存start和text
                temp_start = df.loc[i, 'start']
                temp_text = str(df.loc[i, 'text']) + " " + str(df.loc[i + 2, 'text'])
                temp_interval = float(df.loc[i + 2, 'end']) - float(temp_start)
                
                # 删除n行和n+1行
                df = df.drop(index=[i, i + 1])
                # 重置索引以保持连贯性
                df = df.reset_index(drop=True)
                
                # 替换新的一行的start，并合并text
                df.loc[i, 'start'] = temp_start
                df.loc[i, 'interval'] = temp_interval
                df.loc[i, 'text'] = temp_text
            else:
                i += 1
        
        return df


    def merge_silent_intervals(df):     # 静默时间小于2s的归到上一条字幕
        i = 0
        while i < len(df):
            # 检查当前行的speaker是否为空，且间隔时间小于2秒
            if pd.isnull(df.loc[i, 'speaker']) and df.loc[i, 'interval'] < 2:
                if i > 0:  # 确保不是第一行
                    # 将当前行的间隔时间加到前一行的间隔时间上
                    df.loc[i - 1, 'interval'] += df.loc[i, 'interval']
                    # 当前行的end时间赋值到前一行的end时间上
                    df.loc[i - 1, 'end'] = df.loc[i, 'end']
                    # 删除当前行
                    df = df.drop(index=[i])
                    # 重置索引以保持连贯性
                    df = df.reset_index(drop=True)
                else:
                    # 如果是第一行且符合条件，也删除它，但需要处理合并到下一行的情况
                    df.loc[i + 1, 'start'] = df.loc[i, 'start']
                    df.loc[i + 1, 'interval'] += df.loc[i, 'interval']
                    df = df.drop(index=[i])
                    df = df.reset_index(drop=True)
            else:
                
                i += 1
        
        return df
    

    # 字幕合并：以？。！做结尾，合并后计算字幕长度，大于60不继续合并
    def merge_captions(df):
        # 初始化索引和合并的结果
        i = 0
        while i < len(df) - 1:  # 最后一条不需要检查后面的条目
            current_text = df.loc[i, 'text']
            current_speaker = df.loc[i, 'speaker']
            
            # 确保当前发言者非空
            if pd.notna(current_speaker):
                # 检查文本是否以结束符号结束
                if not current_text.endswith(('？', '。', '！')):
                    # 检查下一条字幕是否为同一个发言者
                    next_speaker = df.loc[i + 1, 'speaker']
                    if current_speaker == next_speaker:
                        # 合并文本和时间，检查合并后长度是否超过60
                        combined_text = current_text + df.loc[i + 1, 'text']
                        if len(combined_text) <= 60:
                            df.loc[i, 'text'] = combined_text
                            df.loc[i, 'end'] = df.loc[i + 1, 'end']
                            df.loc[i, 'interval'] += df.loc[i + 1, 'interval']
                            df = df.drop(index=[i + 1])
                            df = df.reset_index(drop=True)
                            continue  # 继续检查这条字幕是否需要进一步合并
                        else:
                            # 如果合并后长度超过60，不合并
                            pass
            i += 1
        return df


    # 处理端点检测模型失误情况
    def merge3(df):     # 处理端点检测模型失误
        i = 0
        while i < len(df):
            # 当前speaker不为空，speaker说话时长>15，并且说话字数<40的情况
            if pd.notna(df.loc[i, 'speaker']) and df.loc[i, 'interval'] > 15 and len(df.loc[i, 'text']) < 60:
                    new_end_time = df.loc[i, 'end']
                    new_start_time = new_end_time - 15
                    original_start_time = df.loc[i, 'start']
                    
                    # 更新当前行的开始和结束时间
                    df.loc[i, 'start'] = new_start_time
                    df.loc[i, 'interval'] = 15
                    
                    # 创建新的空白行数据
                    new_row = {'start': original_start_time, 'end': new_start_time, 'interval': new_start_time - original_start_time, 'speaker': None, 'text': ''}
                    
                    if i > 0 and pd.isnull(df.loc[i - 1, 'speaker']):
                        # 合并到上一行空白行
                        df.loc[i - 1, 'end'] = new_start_time
                        df.loc[i - 1, 'interval'] += new_row['interval']
                    else:
                        # 在当前行上面插入新行
                        df = pd.concat([df.iloc[:i], pd.DataFrame([new_row]), df.iloc[i:]]).reset_index(drop=True)
                        i += 1  # 调整索引以跳过新插入的行
                    
            i += 1
        
        return df
       


    # 读取CSV文件
    df = pd.read_csv(result_filepath)

    processed_df = process_dataframe1(df)
    processed_df = merge_silent_intervals(processed_df)
    processed_df = merge_captions(processed_df)
    processed_df = merge3(processed_df)
    processed_df = merge_silent_intervals(processed_df)

    processed_df.to_csv(result_filepath, index=False)
    print(f'Processed data has been saved to {result_filepath}')

    # 对结果中的句子进行分类
    speaker_info = [f"{item.get('spk', '')}:{item.get('text', '')}" for item in data]
    question_sentences = classify_sentences(speaker_info)
    
    # 保存分类后的疑问句到 TXT 文件
    questions_filename = f'{audioid}_questions.txt'
    questions_filepath = os.path.join(RESULT_FOLDER, questions_filename)
    with open(questions_filepath, 'w', encoding='utf-8') as txtfile:
        for sentence in question_sentences:
            txtfile.write(sentence + '\n')

    # # 生成SRT内容
    # srt_content = generate_srt(data)
            
     # 生成vtt内容
    vtt_content = generate_vtt(data)

    # 保存SRT到文件
    vtt_filename = f'{audioid}_caption.vtt'
    vtt_filepath = os.path.join(RESULT_FOLDER, vtt_filename)
    with open(vtt_filepath, 'w', encoding='utf-8') as vttfile:
        vttfile.write(vtt_content)
    
    # 处理完成后，更新状态
    audio_processing_status[audioid] = 'completed'

    # 管理结果文件夹中的文件数量
    manage_files(RESULT_FOLDER)

    end_time = time.time()
    print(f"generate time:{end_time-start_time}")


@app.route('/results/<filename>')
def serve_result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/speaker', methods=['POST'])
def upload_and_process_file():
    

    if 'audiofile' not in request.files:
        return jsonify({"code": 5, "msg": "No audio file part"}), 400
    
    file = request.files['audiofile']
    if file.filename == '':
        return jsonify({"code": 5, "msg": "No selected file"}), 400

    audioid = request.form.get('audioid')
    if not audioid:
        return jsonify({"code": 5, "msg": "No audio ID provided"}), 400
    

    audio_filename = f"{audioid}.wav"
    audio_filepath = os.path.join(UPLOAD_FOLDER, audio_filename)

    # 保存文件并管理文件夹中的文件数量
    save_and_manage_file(file, audio_filepath)

    # 初始化audioid的处理状态
    audio_processing_status[audioid] = 'processing'

    # 在后台线程中开始处理，以避免阻塞响应
    thread = threading.Thread(target=process_audio_and_save_result, args=(audio_filepath, audioid))
    thread.start()

    return jsonify({"code": 0, "msg": "Start processing", "data": {"audioid": audioid}})

@app.route('/genstate', methods=['POST'])
def check_generation_status():
    audioid = request.form.get('audioid')
    if not audioid or audioid not in audio_processing_status:
        return jsonify({"code": 5, "msg": "空ID或者不知名ID"}), 400

    if audio_processing_status[audioid] == 'completed':
        # 使用url_for生成结果文件的URL
        result_url = url_for('serve_result_file', filename=f'{audioid}_speakers.csv', _external=True, _scheme='http')
        # 生成疑问句文件的URL
        questions_url = url_for('serve_result_file', filename=f'{audioid}_questions.txt', _external=True, _scheme='http')
        # 生成字幕srt文件的urlsrt
        # 生成疑问句文件的URL
        caption_url = url_for('serve_result_file', filename=f'{audioid}_caption.vtt', _external=True, _scheme='http')
        return jsonify({
            "code": 0, 
            "msg": "processed", 
            "data": {
                "audioid": audioid, 
                "speaker_url": result_url,
                "questions_url": questions_url, # 添加疑问句的url
                "caption_url": caption_url # 添加字幕的url
                }
                })
    elif audio_processing_status[audioid] == 'failed':
        # 如果处理失败，返回失败的状态
        return jsonify({"code": 5, "msg": "音频文件可能缺少声音", "data": {"audioid": audioid}})

    elif audio_processing_status[audioid] == 'nocuda':
        return jsonify({"code": 5, "msg": "显存不足，请稍后重试", "data": {"audioid": audioid}})

    else:
        return jsonify({"code": 1, 
                        "msg": "processing...", 
                        "data": {
                            "audioid": audioid,
                            "speakers_url":"",
                            "questions_url": "",
                            "caption_url": ""
                            }
                            })

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Flask web application.')
    parser.add_argument('--port', type=int, default=8006, help='The port to listen on.')
    parser.add_argument('--gpu', type=str, default='', help='The GPU device to use, e.g., "0" for GPU 0.')
    args = parser.parse_args()
    
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU device {args.gpu}")
    else:
        print("No GPU device specified, using default GPU settings.")
    
    app.run(port=args.port, host='0.0.0.0')
