import typer
from typing import List, Optional
import os
from pathlib import Path

app = typer.Typer()


@app.command()
def main(
        host: str = typer.Option("0.0.0.0", help="Host to bind to"),
        port: int = typer.Option(8000, help="Port to bind to"),
        api_key: Optional[str] = typer.Option(None, help="API key for authentication"),
        config: Optional[Path] = typer.Option(None, help="Path to config file"),
        server_command: Optional[List[str]] = typer.Option(None, help="MCP server command"),
        server_type: Optional[str] = typer.Option(None, help="MCP server type (stdio, sse, streamablehttp)"),
        ssl_certfile: Optional[str] = typer.Option(None, help="SSL certificate file"),
        ssl_keyfile: Optional[str] = typer.Option(None, help="SSL key file"),
        path_prefix: Optional[str] = typer.Option("/", help="Path prefix for API endpoints"),
        workers: Optional[int] = typer.Option(None, help="Number of worker processes (0 for auto)"),
        multi_worker: bool = typer.Option(False, help="Use multi-worker mode with uvicorn"),
):
    """
    Start the MCPO server.
    """
    # 如果指定了多worker模式
    if multi_worker or os.environ.get("MCPO_MULTI_WORKER", "").lower() in ("true", "1", "yes"):
        from mcpo.server import run_server

        # 确定worker数量
        if workers is None:
            if os.environ.get("WORKERS_AUTO", "").lower() in ("true", "1", "yes"):
                workers = 0  # 自动计算
            else:
                try:
                    workers = int(os.environ.get("WORKERS", "0"))
                except (ValueError, TypeError):
                    workers = 0

        # 保存配置到环境变量
        if api_key:
            os.environ["API_KEY"] = api_key
        if config:
            os.environ["CONFIG_PATH"] = str(config)
        if server_type:
            os.environ["SERVER_TYPE"] = server_type
        if server_command:
            os.environ["SERVER_COMMAND"] = " ".join(server_command)

        # 启动多worker服务器
        run_server(
            app="mcpo.main:create_app",
            host=host,
            port=port,
            workers=workers if workers > 0 else None,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            log_level="info"
        )
    else:
        # 传统单worker模式
        import asyncio
        from mcpo.main import run

        kwargs = {}
        if config:
            kwargs["config_path"] = str(config)
        if server_command:
            kwargs["server_command"] = server_command
        if server_type:
            kwargs["server_type"] = server_type
        if ssl_certfile:
            kwargs["ssl_certfile"] = ssl_certfile
        if ssl_keyfile:
            kwargs["ssl_keyfile"] = ssl_keyfile
        if path_prefix:
            kwargs["path_prefix"] = path_prefix

        # 运行单worker服务器
        asyncio.run(
            run(
                host=host,
                port=port,
                api_key=api_key,
                **kwargs
            )
        )


if __name__ == "__main__":
    app()
