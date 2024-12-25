# 接口：8006

# 语音转写，说话人分割

# 环境：jyd/speaker

# paraformer + cam++

# 自动切分左右声道（目前效果最好）
# 带定量清理文件，删除最早的文件（防止内存占满）
# 分离左右声道，左声道无效自动处理右声道

# 处理静默时间 小于2s的归到上一条字幕
# 长时间静默后出现声音，整段时间都会标记到字幕行（瑞华）
# 音频统一处理为16000hz

# 字幕合并：以？。！做结尾，合并后计算字幕长度，大于60不继续合并；
# 处理端点检测模型失误情况 
# 再次处理静默时间 小于2s的归到上一条字幕

# 添加讯飞api
# 带降噪 DeepFilterNet2降噪（切分音频）
# 带显存清理 删除不再使用的对象，减少代码冗余

# 程序保活重启机制 需要修改为绝对路径

# 平均 dBFS (-14.92) 表示文件有较多静音或低音量区域。VAD (Voice Activity Detection)模型配置不适合处理这种音频。

# 优化高并发队列问题


from flask import Flask, request, jsonify, url_for,send_from_directory
import os
import re
import gc
import csv
import time
import json
import argparse
import pandas as pd
from funasr import AutoModel
import threading
import torch
import tempfile
import numpy as np
from pydub import AudioSegment
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from datetime import datetime,timedelta
from pydub import AudioSegment
from app import denoise_audio  # 从app.py导入降噪模块 / 一般不加降噪，降噪会丢失声音细节
from xunfeiapi import RequestApi
from queue import Queue

app = Flask(__name__)

# 创建上传目录、结果目录和缓存目录
UPLOAD_FOLDER = os.path.abspath('./uploads')
RESULT_FOLDER = os.path.abspath('./results')
TEMP_AUDIO_FOLDER = os.path.abspath('./temp_audio_chunks')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)
if not os.path.exists(TEMP_AUDIO_FOLDER):
    os.makedirs(TEMP_AUDIO_FOLDER)
    

# 创建队列用于管理音频处理
audio_processing_queue = Queue()

# 跟踪音频处理状态的字典
audio_processing_status = {}



# 降噪——————DF切分降噪
def process_audio_chunks(input_mp3_path, temp_folder, output_audio_path, audioid, frame_rate=44100):
    print("开始降噪：")
    
    os.makedirs(temp_folder, exist_ok=True)     # 确保临时文件夹存在
    audio_segment = AudioSegment.from_file(file=input_mp3_path).set_frame_rate(frame_rate)  # 将MP3文件转换为WAV格式
    
    chunk_length_ms = 600000
    chunks = [audio_segment[i:i+chunk_length_ms] for i in range(0, len(audio_segment), chunk_length_ms)]  # 将音频切割为10分钟的片段 (600000ms)

    processed_segments = []
    for i, chunk in enumerate(chunks):
        
        chunk_path = os.path.join(temp_folder, f"{audioid}_chunk_{i}.wav")      # 为每个片段生成临时文件路径
        chunk.export(chunk_path, format="wav")
        
        # 对切分出的片段进行降噪处理
        print(f"***************开始降噪第{i+1}段*************")
        denoised_chunk_path = os.path.join(temp_folder, f"{audioid}_denoised_{i}.wav")
        denoise_audio(chunk_path, denoised_chunk_path)
        
        # 加载处理后的片段
        processed_segments.append(AudioSegment.from_file(denoised_chunk_path))
        
        # 清理原始片段文件
        os.remove(chunk_path)
    
    final_segment =  AudioSegment.empty()     # 合并处理后的音频片段
    for segment in processed_segments:
        final_segment += segment
        
    final_segment.export(output_audio_path, format="wav")   # 保存合并后的音频片段

    # 已保存合并后的音频，删除降噪音频片段
    for i in range(len(processed_segments)):
        denoised_chunk_path = os.path.join(temp_folder, f"{audioid}_denoised_{i}.wav")
        os.remove(denoised_chunk_path)
        
    print(f"处理完成，输出文件在：{output_audio_path}")
    

def manage_files(directory, max_files=50):
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
    
    os.makedirs(output_dir, exist_ok=True)   # 确保输出目录存在
    audio = AudioSegment.from_file(input_file)  # 加载音频文件
    audio = audio.set_frame_rate(16000) # 将音频采样率转换为16000Hz
    print("音频采样率降为16000hz")

    # 检查音频是否是立体声
    if audio.channels == 2:
        left_channel = audio.split_to_mono()[0]
        right_channel = audio.split_to_mono()[1]
        
        filename, file_extension = os.path.splitext(os.path.basename(input_file))   # 获取文件名
        # 保存左右声道音频文件
        left_channel_path = os.path.join(output_dir, f"{filename}_left{file_extension}")
        right_channel_path = os.path.join(output_dir, f"{filename}_right{file_extension}")
        left_channel.export(left_channel_path, format="wav")
        right_channel.export(right_channel_path, format="wav")

        print("立体声音频已转换为左声道，保存路径为：", left_channel_path)
        print("立体声音频已转换为右声道，保存路径为：", right_channel_path)
        return left_channel_path, right_channel_path
    else:
        print("原音频是单声道，直接使用原文件。")
        return input_file,None

def classify_sentences(speaker_info):
    print("\n**************************句子分类**************************\n")
    
    
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    question_words_path = os.path.join(current_dir, 'configs/question_words.txt')
    noquestion_words_path = os.path.join(current_dir, 'configs/noquestion_words.txt')
    
    
    # 打开文件
    with open(question_words_path, 'r', encoding='utf-8') as file:
        question_words = [word.strip() for word in file.readlines()]
    
    with open(noquestion_words_path, 'r', encoding='utf-8') as file:
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

def initialize_model():
    """初始化模型并返回实例。"""
    
    
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
    
    
    return model

def clear_cuda_cache():
    """清理CUDA显存和Python垃圾回收。"""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def process_audio_and_save_result(audio_filepath, audioid, audiodenoise):
    
    start_time = time.time()
    print("start time: 开始计时")
    res=[]
    left_processed = False
    
    try:
        # 设置保存结果CSV文件地址
        result_filename = f'{audioid}_speakers.csv'
        result_filepath = os.path.join(RESULT_FOLDER, result_filename)

        input_mp3_path = audio_filepath
        temp_folder = 'temp_audio_chunks'  # 用于存储音频片段的临时文件夹
        output_audio_path = f'temp_audio_chunks/{audioid}_denoised.wav'  # 指定输出文件的路径

        # 将音频转换为左声道
        left_channel_path,right_channel_path = convert_audio_to_left_channel(input_mp3_path, temp_folder)

        # 切分音频降噪处理
        if audiodenoise == "1":
            process_audio_chunks(left_channel_path, temp_folder, output_audio_path,audioid)
        else:
            output_audio_path = left_channel_path
        
        print(f"开始处理{output_audio_path}音频文件")
        
        clear_cuda_cache()
        
        model = initialize_model()
        
        res = model.generate(input = output_audio_path, batch_size_s=300, hotword = '基因')
        
        
        # 检查res列表是否为空,如果为空则开始处理右声道
        if not res or "sentence_info" not in res[0]:
            print("没有从左声道音频文件中识别出任何内容")
        else:
            clear_cuda_cache()
            if os.path.exists(left_channel_path):
                os.remove(left_channel_path)
            left_processed = True
            
            del model,left_channel_path,output_audio_path
        
    except Exception as e:
        if "CUDA out of memory" in str(e):
            audio_processing_status[audioid] = 'nocuda'     # 更新状态为nocuda
            print(f"处理错误可能是显存不足，异常信息为：{e}")
        else:
            print(f"处理左声道音频失败，异常信息为：{e}")
            
        clear_cuda_cache()

    finally:
        clear_cuda_cache()
        
    if not left_processed:      # 左声道不行得话，处理右声道
        try:
            # 切分音频降噪处理
            if audiodenoise == "1":
                process_audio_chunks(right_channel_path, temp_folder, output_audio_path,audioid)
            else:
                output_audio_path = right_channel_path
            
            print(f"开始处理{output_audio_path}音频文件")
            
            clear_cuda_cache()

            model = initialize_model()
            
            res = model.generate(input = output_audio_path, batch_size_s=300, hotword = '基因')

            clear_cuda_cache()
            
            if os.path.exists(right_channel_path):
                os.remove(right_channel_path)
            
            del model,right_channel_path,output_audio_path
                
        except Exception as e:
            audio_processing_status[audioid] = 'failed'  # 更新状态为失败
            print(f"处理音频文件错误，异常信息为: {e}")
            clear_cuda_cache()
        finally:
            clear_cuda_cache()
            
    # 检查res列表是否为空
    if not res or "sentence_info" not in res[0]:
        print("没有从音频文件中识别出任何内容")
        audio_processing_status[audioid] = 'failed'  # 更新状态为失败
        return  
    
    # 打印出文本
    data = res[0]["sentence_info"]
 
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
        while i < len(df) - 1:  # 修改条件以避免索引越界，检查当前行和下一行
            # 检查当前行和下一行的speaker是否相等且不为空，且text长度小于等于6
            if pd.notnull(df.loc[i, 'speaker']) and \
                pd.notnull(df.loc[i + 1, 'speaker']) and \
                df.loc[i, 'speaker'] == df.loc[i + 1, 'speaker'] and \
                len(str(df.loc[i, 'text'])) <= 6:
                
                    # 暂存start和text
                    temp_start = df.loc[i, 'start']
                    temp_text = str(df.loc[i, 'text']) + " " + str(df.loc[i + 1, 'text'])
                    temp_interval = float(df.loc[i + 1, 'end']) - float(temp_start)
                    
                    # 删除当前行和下一行
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

            
    # 生成vtt内容
    vtt_content = generate_vtt(data)


    # 保存vtt到文件
    vtt_filename = f'{audioid}_caption.vtt'
    vtt_filepath = os.path.join(RESULT_FOLDER, vtt_filename)
    with open(vtt_filepath, 'w', encoding='utf-8') as vttfile:
        vttfile.write(vtt_content)
    
    # 处理完成后，更新状态
    audio_processing_status[audioid] = 'completed'

    # 管理结果文件夹中的文件数量
    manage_files(RESULT_FOLDER)
    # 管理临时文件夹中的文件数量（左声道，降噪，切分）
    manage_files(temp_folder)
    
    clear_cuda_cache()

    end_time = time.time()
    print(f"generate time:{end_time-start_time}")

def process_audio_and_save_result_xunfei(audio_filepath, audioid):
    start_time = time.time()
    print("开始调用讯飞接口")
    audioid = audioid
    # 保存结果到 CSV 文件
    result_filename = f'{audioid}_speakers.csv'
    result_filepath = os.path.join(RESULT_FOLDER, result_filename)
    
    result = RequestApi(upload_file_path= audio_filepath).get_result()

    # 先解析 'orderResult' 的 JSON 字符串
    order_result_str = result['content']['orderResult']

    # 将字符串解析为 JSON 对象
    try:
        order_result_json = json.loads(order_result_str)
    except json.JSONDecodeError as e:
        print(f"解析 JSON 时出错: {e}")
        exit(1)

    # 现在可以访问 lattice2 等字段
    lattice2 = order_result_json.get('lattice2')
    res = []
    for item in lattice2:
        # 获取说话人信息
        speaker = item.get('spk', '未知说话人')
        # speaker取到“段落-”字符
        speaker = int(speaker.split('段落-')[1])-1

        # 获取时间戳
        begin_time = int(item.get('begin', '未知时间戳'))/1000
        end_time = int(item.get('end', '未知时间戳'))/1000
        interval = end_time - begin_time
        # 解析出文字片段的内容
        json_1best = item['json_1best']
        text = []
        for rt in json_1best['st']['rt']:
            for ws in rt['ws']:
                # 提取每个片段的文字
                text.append(''.join([cw['w'] for cw in ws['cw']]))
        res.append((speaker, ''.join(text),begin_time,end_time,interval))
    
    # 将结果保存到 txt 文件中
    with open(result_filepath, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'interval', 'speaker', 'text'])
        for i, (speaker, speech,begin_time,end_time,interval) in enumerate(res, 1):
            writer.writerow([begin_time, end_time, interval, speaker, speech])
    


    def merge_silent_intervals(df):     # 静默时间小于2s的归到上一条字幕
        i = 0
        while i < len(df)-1:
            # 检查当前行的speaker和下一行speaker相等，且间隔时间小于2秒
            if df.loc[i, 'speaker'] == df.loc[i + 1, 'speaker'] and df.loc[i, 'interval'] < 1:
                if i > 0:  # 确保不是第一行
                    # 将当前行的间隔时间加到前一行的间隔时间上
                    df.loc[i, 'interval'] += df.loc[i+1, 'interval']
                    # 当前行的end时间赋值到前一行的end时间上
                    df.loc[i, 'end'] = df.loc[i+1, 'end']
                    # 当前行的内容加到上一行
                    df.loc[i, 'text'] += df.loc[i+1, 'text']
                    # 删除当前行
                    df = df.drop(index=[i+1])
                    # 重置索引以保持连贯性
                    df = df.reset_index(drop=True)
                else:
                    # 如果是第一行且符合条件，也删除它，但需要处理合并到下一行的情况
                    df.loc[i + 1, 'start'] = df.loc[i, 'start']
                    df.loc[i + 1, 'interval'] += df.loc[i, 'interval']
                    df.loc[i + 1, 'text'] = df.loc[i, 'text']
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

    
    processed_df = merge_captions(df)
    processed_df = merge_silent_intervals(processed_df)
    processed_df = merge3(processed_df)



    processed_df.to_csv(result_filepath, index=False)

    

    def convert_to_vtt_time(seconds):
        """Convert seconds to VTT time format (hh:mm:ss.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02}:{minutes:02}:{int(seconds):02}.{milliseconds:03}"


    def convert_csv_to_vtt(csv_file, output_vtt):
        """Convert a CSV file to VTT format and save it."""
        # Load the CSV
        df = pd.read_csv(csv_file)
        
        # Open the output VTT file
        with open(output_vtt, 'w', encoding='utf-8') as vtt_file:
            # Write the WebVTT header
            vtt_file.write("WEBVTT\n\n")
            
            # Iterate through each row and format the subtitle block
            for index, row in df.iterrows():
                start_time = convert_to_vtt_time(row['start'])
                end_time = convert_to_vtt_time(row['end'])
                speaker = f"speaker{row['speaker']}"
                text = row['text']
                
                # Write the VTT block
                vtt_file.write(f"{start_time} --> {end_time}\n{speaker}: {text}\n\n")

    result_vttname = f'{audioid}_caption.vtt'
    result_vttpath = os.path.join(RESULT_FOLDER, result_vttname)
    # 生成VTT文件
    convert_csv_to_vtt(result_filepath, result_vttpath)

    # 管理结果文件夹中的文件数量
    manage_files(RESULT_FOLDER)


    print(f'Processed data has been saved to {result_filepath}')



    # 处理完成后，更新状态
    audio_processing_status[audioid] = 'completed'
    end_time = time.time()
    print(f"generate time:{end_time-start_time}")


def audio_processing_worker():
    while True:
        # 获取队列中的音频文件
        audioid, audio_filepath, audiodenoise = audio_processing_queue.get()
        if audioid is None:
            break  # 如果是终止信号，则退出线程
        process_audio_and_save_result(audio_filepath, audioid, audiodenoise)
        audio_processing_queue.task_done()  # 标记任务完成

# 启动工作线程，最多可同时处理1个音频
worker_thread = threading.Thread(target=audio_processing_worker, daemon=True)
worker_thread.start()



@app.route('/results/<filename>')       #提供一个 Flask 路由，用于返回存储在 RESULT_FOLDER 目录下的文件。当用户请求该路由时，Flask 会尝试从指定的结果文件夹中发送对应的文件。
def serve_result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/speaker', methods=['POST'])
def upload_and_process_file():
    

    if 'audiofile' not in request.files:
        return jsonify({"code": 5, "msg": "No audio file part"}), 200
    
    file = request.files['audiofile']
    if file.filename == '':
        return jsonify({"code": 5, "msg": "No selected file"}), 200

    audioid = request.form.get('audioid')
    audiotype = request.form.get('audiotype')
    audiodenoise = request.form.get('audiodenoise')
    
    if not audioid:
        return jsonify({"code": 5, "msg": "No audio ID provided"}), 200
    

    audio_filename = f"{audioid}.wav"
    audio_filepath = os.path.join(UPLOAD_FOLDER, audio_filename)

    # 保存文件并管理文件夹中的文件数量
    save_and_manage_file(file, audio_filepath)

    # 初始化audioid的处理状态
    audio_processing_status[audioid] = 'processing'

    if audiotype == "1":
        # 在后台线程中开始处理，以避免阻塞响应
        thread = threading.Thread(target=process_audio_and_save_result_xunfei, args=(audio_filepath, audioid))
        thread.start()
    else:
       # 将音频文件加入队列进行处理
        audio_processing_queue.put((audioid, audio_filepath, audiodenoise))
    
    return jsonify({"code": 0, "msg": "Start processing", "data": {"audioid": audioid}})

@app.route('/genstate', methods=['POST'])
def check_generation_status():
    audioid = request.form.get('audioid')
    if not audioid or audioid not in audio_processing_status:
        return jsonify({"code": 5, "msg": "Empty ID or unknown ID"}), 200

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
        return jsonify({"code": 5, "msg": "The audio file may be missing sound", "data": {"audioid": audioid}})

    elif audio_processing_status[audioid] == 'nocuda':
        return jsonify({"code": 5, "msg": "CUDA is out of memory, please try again later", "data": {"audioid": audioid}})

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
    parser.add_argument('--port', type=int, default=8006, help='端口号')
    args = parser.parse_args()
    
    app.run(port=args.port, host='0.0.0.0')
