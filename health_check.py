# 程序健康保活机制

# 设置超时时间20秒

# 最大连续超时次数3次


import datetime
import requests
import os

# 配置 Flask 服务的 URL 和测试参数
SERVICES = [
    {"url": "http://localhost:8006/genstate", "params": {"audioid": "test"}},
    {"url": "http://localhost:8003/keywords", "params": {"text": "测试文本"}}
]

# 配置重启命令
SERVICE_RESTART_COMMANDS = {
    8006: {
        "conda_env_path": "/home/wzhpc/anaconda3/envs/jyd/bin/python",
        "script_path": "/opt/jyd01/wangruihua/api/speaker/flask_speaker.py",
        "cuda_device": "3"
    },
    8003: {
        "conda_env_path": "/home/wzhpc/anaconda3/envs/jyd/bin/python",
        "script_path": "/opt/jyd01/wangruihua/api/speaker/flask_keywords.py",
        "cuda_device": "0"
    }
}

# 配置日志目录
LOG_DIR = "/opt/jyd01/wangruihua/api/speaker/logs" 

# 设置超时时间和最大重试次数
TIMEOUT = 20  # 单次请求超时时间（秒）
MAX_RETRIES = 3  # 最大连续超时次数


def check_service(url, params):
    """检查服务是否正常运行"""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.post(url, json=params, timeout=TIMEOUT)
            if response.status_code == 200:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"服务 {url} 正常——————{current_time}")
                return True
            else:
                print(f"服务 {url} 返回错误状态码: {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            retries += 1
            print(f"服务 {url} 请求超时，第 {retries} 次重试...")
        except Exception as e:
            print(f"服务 {url} 异常: {e}")
            return False
    print(f"服务 {url} 超时达到最大重试次数，判定为异常")
    return False


def kill_service(script_name):
    """终止指定脚本的进程"""
    try:
        kill_command = f"pkill -f {script_name}"        # 这块可能有问题，貌似杀不掉
        exit_code = os.system(kill_command)
        if exit_code == 0:
            print(f"已终止运行的服务：{script_name}")
        else:
            print(f"未找到运行中的服务：{script_name}")
    except Exception as e:
        print(f"终止服务时发生异常: {e}")


def validate_paths(port):
    """验证关键路径是否存在"""
    restart_info = SERVICE_RESTART_COMMANDS[port]
    conda_env_path = restart_info["conda_env_path"]
    script_path = restart_info["script_path"]

    if not os.path.exists(conda_env_path):
        raise FileNotFoundError(f"Conda 环境路径不存在：{conda_env_path}")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"脚本路径不存在：{script_path}")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"日志目录不存在，已创建：{LOG_DIR}")


def restart_service(port):
    """重启服务"""
    try:
        restart_info = SERVICE_RESTART_COMMANDS[port]
        conda_env_path = restart_info["conda_env_path"]
        script_path = restart_info["script_path"]
        cuda_device = restart_info["cuda_device"]

        # 构造重启命令
        restart_command = (
            f"CUDA_VISIBLE_DEVICES={cuda_device} "
            f"{conda_env_path} {script_path} --port {port}"
        )
        
        # 终止之前运行的服务
        kill_service(os.path.basename(script_path))
        
        log_path = os.path.join(LOG_DIR, f"nohup_{port}.log")
        print(f"尝试重启服务: {restart_command}")
        
        # 启动服务
        os.system(f"nohup bash -c '{restart_command}' >> {log_path} 2>&1 &")
        print(f"服务 {port} 已重启，日志记录到: {log_path}")
    except Exception as e:
        print(f"重启服务 {port} 时出现异常: {e}")


# 检查每个服务并根据情况重启
for service in SERVICES:
    url = service["url"]
    params = service["params"]
    port = int(url.split(":")[2].split("/")[0])  # 从 URL 中提取端口号
    if not check_service(url, params):
        print(f"服务 {port} 异常，尝试重启...")
        validate_paths(port)
        restart_service(port)
