import json
import os
import logging
import socket
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount

logger = logging.getLogger(__name__)

from mcpo.utils.main import get_model_fields, get_tool_handler
from mcpo.utils.auth import get_verify_api_key, APIKeyMiddleware


# 添加中间件来识别请求的worker
# 生成一个唯一的worker ID
WORKER_ID = str(uuid.uuid4())[:8]
# 获取主机名，用于标识worker
HOSTNAME = socket.gethostname()

# 使用标准的BaseHTTPMiddleware实现worker标识中间件
class WorkerIdentificationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # 添加标识worker的头部
        response.headers["X-Worker-ID"] = WORKER_ID
        response.headers["X-Worker-Hostname"] = HOSTNAME
        return response

async def create_dynamic_endpoints(app: FastAPI, api_dependency=None):
    # 现有代码保持不变
    # ...
    session: ClientSession = app.state.session
    if not session:
        raise ValueError("Session is not initialized in the app state.")

    result = await session.initialize()
    server_info = getattr(result, "serverInfo", None)
    if server_info:
        app.title = server_info.name or app.title
        app.description = (
            f"{server_info.name} MCP Server" if server_info.name else app.description
        )
        app.version = server_info.version or app.version

    tools_result = await session.list_tools()
    tools = tools_result.tools

    for tool in tools:
        endpoint_name = tool.name
        endpoint_description = tool.description

        inputSchema = tool.inputSchema
        outputSchema = getattr(tool, "outputSchema", None)

        form_model_fields = get_model_fields(
            f"{endpoint_name}_form_model",
            inputSchema.get("properties", {}),
            inputSchema.get("required", []),
            inputSchema.get("$defs", {}),
        )

        response_model_fields = None
        if outputSchema:
            response_model_fields = get_model_fields(
                f"{endpoint_name}_response_model",
                outputSchema.get("properties", {}),
                outputSchema.get("required", []),
                outputSchema.get("$defs", {}),
            )

        tool_handler = get_tool_handler(
            session,
            endpoint_name,
            form_model_fields,
            response_model_fields,
        )

        app.post(
            f"/{endpoint_name}",
            summary=endpoint_name.replace("_", " ").title(),
            description=endpoint_description,
            response_model_exclude_none=True,
            dependencies=[Depends(api_dependency)] if api_dependency else [],
        )(tool_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 现有代码保持不变
    # ...
    server_type = getattr(app.state, "server_type", "stdio")
    command = getattr(app.state, "command", None)
    args = getattr(app.state, "args", [])
    env = getattr(app.state, "env", {})

    args = args if isinstance(args, list) else [args]
    api_dependency = getattr(app.state, "api_dependency", None)

    if (server_type == "stdio" and not command) or (
            server_type == "sse" and not args[0]
    ):
        # Main app lifespan (when config_path is provided)
        async with AsyncExitStack() as stack:
            for route in app.routes:
                if isinstance(route, Mount) and isinstance(route.app, FastAPI):
                    await stack.enter_async_context(
                        route.app.router.lifespan_context(route.app),  # noqa
                    )
            yield
    else:
        if server_type == "stdio":
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env={**env},
            )

            async with stdio_client(server_params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    await create_dynamic_endpoints(app, api_dependency=api_dependency)
                    yield
        if server_type == "sse":
            async with sse_client(url=args[0]) as (
                    reader,
                    writer,
            ):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    await create_dynamic_endpoints(app, api_dependency=api_dependency)
                    yield
        if server_type == "streamablehttp" or server_type == "streamable_http":
            # Ensure URL has trailing slash to avoid redirects
            url = args[0]
            if not url.endswith("/"):
                url = f"{url}/"

            # Connect using streamablehttp_client from the SDK, similar to sse_client
            async with streamablehttp_client(url=url) as (
                    reader,
                    writer,
                    _,  # get_session_id callback not needed for ClientSession
            ):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    await create_dynamic_endpoints(app, api_dependency=api_dependency)
                    yield


def create_app(**kwargs):
    """创建FastAPI应用实例，为多worker模式服务"""
    # 从环境变量或kwargs获取配置
    host = kwargs.get("host", "127.0.0.1")
    port = kwargs.get("port", 8000)
    api_key = kwargs.get("api_key", os.environ.get("API_KEY", ""))
    cors_allow_origins = kwargs.get("cors_allow_origins", ["*"])
    server_type = kwargs.get("server_type")
    server_command = kwargs.get("server_command")
    config_path = kwargs.get("config_path", os.environ.get("CONFIG_PATH"))
    name = kwargs.get("name", "MCP OpenAPI Proxy")
    description = kwargs.get("description", "Automatically generated API from MCP Tool Schemas")
    version = kwargs.get("version", "1.0")
    ssl_certfile = kwargs.get("ssl_certfile")
    ssl_keyfile = kwargs.get("ssl_keyfile")
    path_prefix = kwargs.get("path_prefix", "/")
    strict_auth = kwargs.get("strict_auth", False)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Creating MCPO Application...")
    logger.info(f"  Name: {name}")
    logger.info(f"  Version: {version}")
    logger.info(f"  API Key: {'Provided' if api_key else 'Not Provided'}")

    # 创建API依赖
    api_dependency = get_verify_api_key(api_key) if api_key else None

    # 创建主应用
    main_app = FastAPI(
        title=name,
        description=description,
        version=version,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        lifespan=lifespan,
    )
    main_app.middleware("http")(WorkerIdentificationMiddleware())

    main_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 添加中间件以保护文档和规范
    if api_key and strict_auth:
        main_app.add_middleware(APIKeyMiddleware, api_key=api_key)

    # 配置服务器
    if server_type == "sse":
        logger.info(
            f"Configuring for a single SSE MCP Server with URL {server_command[0]}"
        )
        main_app.state.server_type = "sse"
        main_app.state.args = server_command[0]  # Expects URL as the first element
        main_app.state.api_dependency = api_dependency
    elif server_type == "streamablehttp" or server_type == "streamable_http":
        logger.info(
            f"Configuring for a single StreamableHTTP MCP Server with URL {server_command[0]}"
        )
        main_app.state.server_type = "streamablehttp"
        main_app.state.args = server_command[0]  # Expects URL as the first element
        main_app.state.api_dependency = api_dependency
    elif server_command:  # This handles stdio
        logger.info(
            f"Configuring for a single Stdio MCP Server with command: {' '.join(server_command)}"
        )
        main_app.state.server_type = "stdio"  # Explicitly set type
        main_app.state.command = server_command[0]
        main_app.state.args = server_command[1:]
        main_app.state.env = os.environ.copy()
        main_app.state.api_dependency = api_dependency
    elif config_path:
        logger.info(f"Loading MCP server configurations from: {config_path}")
        with open(config_path, "r") as f:
            config_data = json.load(f)

        mcp_servers = config_data.get("mcpServers", {})
        if not mcp_servers:
            logger.error(f"No 'mcpServers' found in config file: {config_path}")
            raise ValueError("No 'mcpServers' found in config file.")

        logger.info("Configured MCP Servers:")
        for server_name_cfg, server_cfg_details in mcp_servers.items():
            if server_cfg_details.get("command"):
                args_info = (
                    f" with args: {server_cfg_details['args']}"
                    if server_cfg_details.get("args")
                    else ""
                )
                logger.info(
                    f"  Configuring Stdio MCP Server '{server_name_cfg}' with command: {server_cfg_details['command']}{args_info}"
                )
            elif server_cfg_details.get("type") == "sse" and server_cfg_details.get(
                    "url"
            ):
                logger.info(
                    f"  Configuring SSE MCP Server '{server_name_cfg}' with URL: {server_cfg_details['url']}"
                )
            elif (
                    server_cfg_details.get("type") == "streamablehttp"
                    or server_cfg_details.get("type") == "streamable_http"
            ) and server_cfg_details.get("url"):
                logger.info(
                    f"  Configuring StreamableHTTP MCP Server '{server_name_cfg}' with URL: {server_cfg_details['url']}"
                )
            elif server_cfg_details.get("url"):  # Fallback for old SSE config
                logger.info(
                    f"  Configuring SSE (fallback) MCP Server '{server_name_cfg}' with URL: {server_cfg_details['url']}"
                )
            else:
                logger.warning(
                    f"  Unknown configuration for MCP server: {server_name_cfg}"
                )

        main_app.description += "\n\n- **available tools**："
        for server_name, server_cfg in mcp_servers.items():
            sub_app = FastAPI(
                title=f"{server_name}",
                description=f"{server_name} MCP Server\n\n- [back to tool list](/docs)",
                version="1.0",
                lifespan=lifespan,
            )
            sub_app.middleware("http")(WorkerIdentificationMiddleware())

            sub_app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_allow_origins or ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            if server_cfg.get("command"):
                # stdio
                sub_app.state.server_type = "stdio"
                sub_app.state.command = server_cfg["command"]
                sub_app.state.args = server_cfg.get("args", [])
                sub_app.state.env = {**os.environ, **server_cfg.get("env", {})}

            server_config_type = server_cfg.get("type")
            if server_config_type == "sse" and server_cfg.get("url"):
                sub_app.state.server_type = "sse"
                sub_app.state.args = server_cfg["url"]
            elif (
                    server_config_type == "streamablehttp"
                    or server_config_type == "streamable_http"
            ) and server_cfg.get("url"):
                # Store the URL with trailing slash to avoid redirects
                url = server_cfg["url"]
                if not url.endswith("/"):
                    url = f"{url}/"
                sub_app.state.server_type = "streamablehttp"
                sub_app.state.args = url
            elif not server_config_type and server_cfg.get(
                    "url"
            ):  # Fallback for old SSE config
                sub_app.state.server_type = "sse"
                sub_app.state.args = server_cfg["url"]

            # Add middleware to protect also documentation and spec
            if api_key and strict_auth:
                sub_app.add_middleware(APIKeyMiddleware, api_key=api_key)

            sub_app.state.api_dependency = api_dependency

            main_app.mount(f"{path_prefix}{server_name}", sub_app)
            main_app.description += f"\n    - [{server_name}](/{server_name}/docs)"
    else:
        logger.error("MCPO server_command or config_path must be provided.")
        raise ValueError("You must provide either server_command or config.")

    return main_app


# 保留原始的run函数，但将其作为单worker模式的入口点
async def run(
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: Optional[str] = "",
        cors_allow_origins=["*"],
        **kwargs,
):
    """单worker模式的入口点，保持向后兼容性"""
    app = create_app(
        host=host,
        port=port,
        api_key=api_key,
        cors_allow_origins=cors_allow_origins,
        **kwargs
    )

    logger.info("Uvicorn server starting in single-worker mode...")
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        ssl_certfile=kwargs.get("ssl_certfile"),
        ssl_keyfile=kwargs.get("ssl_keyfile"),
        log_level="info",
    )
    server = uvicorn.Server(config)

    await server.serve()
