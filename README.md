# 课堂质量诊断

## 新建虚拟环境

```python
conda create -n speaker python=3.8
conda activate speaker
```

## 安装依赖包

```python
pip install -r requirements.txt
```

## 运行：

### 关键字服务接口

```python
nohup python flask_keywords.py & tail -f nohup.out
```

### 说话人分割服务接口

```python
nohup python flask_speaker.py & tail -f nohup.out
```

### 语音转写web界面（常态化运行用于支持其他业务）
```python
nohup python gradio_asr.py & tail -f nohup.out
```
