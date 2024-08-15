# speaker

工作项目——课堂质量诊断

### 运行：

```python
pip install -r requirements.txt
```

### 关键字服务接口
```python
nohup python flask_keyword.py & tail -f nohup.out
```

### 说话人分割服务接口
```python
nohup python flask_speaker.py & tail -f nohup.out
```
