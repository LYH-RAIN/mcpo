import os
import multiprocessing
import uvicorn
import argparse
import json
from typing import Dict, Any, Optional


def get_worker_count() -> int:
    """自动确定worker数量，基于CPU核心数"""
    if os.environ.get("WORKERS_AUTO", "").lower() in ("true", "1", "yes"):
        # 使用可用CPU核心数，但至少使用2个worker
        return max(multiprocessing.cpu_count(), 2)

    # 尝试从环境变量获取指定的worker数
    try:
        workers = int(os.environ.get("WORKERS", "0"))
        if workers > 0:
            return workers
    except (ValueError, TypeError):
        pass

    # 默认返回2个worker
    return 2


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}


def run_server(app: str, host: str = "0.0.0.0", port: int = 8000,
               workers: int = None, ssl_certfile: str = None,
               ssl_keyfile: str = None, log_level: str = "info"):
    """使用Uvicorn启动多worker服务器"""
    if workers is None:
        workers = get_worker_count()

    print(f"Starting server with {workers} workers on {host}:{port}")

    uvicorn_config = {
        "app": app,
        "host": host,
        "port": port,
        "workers": workers,
        "log_level": log_level,
    }

    if ssl_certfile:
        uvicorn_config["ssl_certfile"] = ssl_certfile
    if ssl_keyfile:
        uvicorn_config["ssl_keyfile"] = ssl_keyfile

    uvicorn.run(**uvicorn_config)


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description="Run MCPO server with multiple workers")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--ssl-certfile", help="SSL certificate file")
    parser.add_argument("--ssl-keyfile", help="SSL key file")
    parser.add_argument("--log-level", default="info", help="Logging level")

    args = parser.parse_args()

    # 设置环境变量，供应用程序读取
    if args.api_key:
        os.environ["API_KEY"] = args.api_key

    # 构建应用程序路径字符串
    app_path = "mcpo.main:create_app"

    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    run_server(
        app=app_path,
        host=args.host,
        port=args.port,
        workers=args.workers,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
