import json
import traceback
import os
import logging
from typing import Any, Dict, ForwardRef, List, Optional, Type, Union, get_origin, get_args

from fastapi import HTTPException

from mcp import ClientSession, types
from mcp.types import (
    CallToolResult,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)

from mcp.shared.exceptions import McpError

from pydantic import Field, create_model
from pydantic.fields import FieldInfo

# 导入响应适配器
from .response_adapters import AutoResponseAdapter

logger = logging.getLogger(__name__)

MCP_ERROR_TO_HTTP_STATUS = {
    PARSE_ERROR: 400,
    INVALID_REQUEST: 400,
    METHOD_NOT_FOUND: 404,
    INVALID_PARAMS: 422,
    INTERNAL_ERROR: 500,
}

def process_tool_response(result: CallToolResult) -> list:
    """Universal response processor for all tool endpoints"""
    response = []
    for content in result.content:
        if isinstance(content, types.TextContent):
            text = content.text
            if isinstance(text, str):
                try:
                    text = json.loads(text)
                except json.JSONDecodeError:
                    pass
            response.append(text)
        elif isinstance(content, types.ImageContent):
            image_data = f"data:{content.mimeType};base64,{content.data}"
            response.append(image_data)
        elif isinstance(content, types.EmbeddedResource):
            # TODO: Handle embedded resources
            response.append("Embedded resource not supported yet.")
    return response

def smart_response_adapter(response_data, response_model, endpoint_name: str):
    """
    智能响应适配器：使用方案四的自动类型推断
    """
    logger.info(f"🔧 [ADAPTER] smart_response_adapter called for {endpoint_name}")
    logger.info(f"🔧 [ADAPTER] response_data type: {type(response_data)}, response_model: {response_model}")

    # 使用 AutoResponseAdapter 进行自动适配
    result = AutoResponseAdapter.auto_adapt(response_data, response_model, endpoint_name)

    logger.info(f"🔧 [ADAPTER] {endpoint_name}: adaptation completed")
    return result


def convert_list_items(data_list: list, target_type: type):
    """
    转换列表中的项目到目标类型
    """
    converted_list = []

    for item in data_list:
        converted_item = convert_to_basic_type(item, target_type)
        converted_list.append(converted_item)

    return converted_list


def convert_to_basic_type(item, target_type: type):
    """
    将项目转换为基本类型 - 增强版本，支持更智能的转换
    """
    try:
        if target_type == str:
            if isinstance(item, dict):
                return extract_string_from_dict(item)
            elif isinstance(item, (list, tuple)):
                # 如果是列表或元组，尝试提取第一个有意义的元素
                if item:
                    return convert_to_basic_type(item[0], str)
                return "[]"
            else:
                return str(item)
        elif target_type == int:
            if isinstance(item, str):
                # 尝试从字符串中提取数字
                import re
                numbers = re.findall(r'\d+', item)
                return int(numbers[0]) if numbers else 0
            elif isinstance(item, dict):
                # 从字典中尝试提取数字字段
                for key in ['id', 'number', 'count', 'index']:
                    if key in item and isinstance(item[key], (int, str)):
                        try:
                            return int(item[key])
                        except (ValueError, TypeError):
                            continue
                return 0
            return int(item) if isinstance(item, (int, float)) else 0
        elif target_type == float:
            if isinstance(item, str):
                import re
                numbers = re.findall(r'\d+\.?\d*', item)
                return float(numbers[0]) if numbers else 0.0
            elif isinstance(item, dict):
                # 从字典中尝试提取数字字段
                for key in ['value', 'amount', 'price', 'score']:
                    if key in item and isinstance(item[key], (int, float, str)):
                        try:
                            return float(item[key])
                        except (ValueError, TypeError):
                            continue
                return 0.0
            return float(item) if isinstance(item, (int, float)) else 0.0
        elif target_type == bool:
            if isinstance(item, str):
                return item.lower() in ('true', '1', 'yes', 'on', 'enabled', 'active')
            elif isinstance(item, dict):
                # 从字典中尝试提取布尔字段
                for key in ['enabled', 'active', 'valid', 'success', 'status']:
                    if key in item:
                        return bool(item[key])
                return bool(item)  # 非空字典为 True
            return bool(item)
        else:
            return item
    except (ValueError, TypeError):
        # 如果转换失败，返回默认值
        if target_type == str:
            return str(item)
        elif target_type == int:
            return 0
        elif target_type == float:
            return 0.0
        elif target_type == bool:
            return False
        else:
            return item


def extract_string_from_dict(item_dict: dict) -> str:
    """
    从字典中智能提取字符串表示
    """
    # 优先级顺序的字段名
    priority_fields = ['name', 'title', 'fullname', 'id', 'key', 'label', 'display_name', 'description']

    for field in priority_fields:
        if field in item_dict and isinstance(item_dict[field], str) and item_dict[field].strip():
            return item_dict[field]

    # 如果没有找到优先字段，寻找第一个非空字符串值
    for key, value in item_dict.items():
        if isinstance(value, str) and value.strip():
            return value

    # 寻找第一个数字并转为字符串
    for key, value in item_dict.items():
        if isinstance(value, (int, float)):
            return str(value)

    # 最后的备选方案：JSON 字符串
    try:
        return json.dumps(item_dict, ensure_ascii=False, separators=(',', ':'))
    except:
        return str(item_dict)


def name_needs_alias(name: str) -> bool:
    """Check if a field name needs aliasing (for now if it starts with '__')."""
    return name.startswith('__')


def generate_alias_name(original_name: str, existing_names: set) -> str:
    """
    Generate an alias field name by stripping unwanted chars, and avoiding conflicts with existing names.

    Args:
        original_name: The original field name (should start with '__')
        existing_names: Set of existing names to avoid conflicts with

    Returns:
        An alias name that doesn't conflict with existing names
    """
    alias_name = original_name.lstrip('_')
    # Handle potential naming conflicts
    original_alias_name = alias_name
    suffix_counter = 1
    while alias_name in existing_names:
        alias_name = f"{original_alias_name}_{suffix_counter}"
        suffix_counter += 1
    return alias_name


def _process_schema_property(
        _model_cache: Dict[str, Type],
        prop_schema: Dict[str, Any],
        model_name_prefix: str,
        prop_name: str,
        is_required: bool,
        schema_defs: Optional[Dict] = None,
        server_name: str = "unknown",
) -> tuple[Union[Type, List, ForwardRef, Any], FieldInfo]:
    """
    Recursively processes a schema property to determine its Python type hint
    and Pydantic Field definition.

    Returns:
        A tuple containing (python_type_hint, pydantic_field).
        The pydantic_field contains default value and description.
    """
    if "$ref" in prop_schema:
        ref = prop_schema["$ref"]
        if ref.startswith("#/properties/"):
            # Remove common prefix in pathes.
            prefix_path = model_name_prefix.split("_form_model_")[-1]
            ref_path = ref.split("#/properties/")[-1]
            # Translate $ref path to model_name_prefix style.
            ref_path = ref_path.replace("/properties/", "_model_")
            ref_path = ref_path.replace("/items", "_item")
            # If $ref path is a prefix substring of model_name_prefix path,
            # there exists a circular reference.
            # The loop should be broke with a return to avoid exception.
            if prefix_path.startswith(ref_path):
                # TODO: Find the exact type hint for the $ref.
                return Any, Field(default=None, description="")
        original_ref = ref
        ref = ref.split
        if schema_defs is None or ref not in schema_defs:
            logger.warning(f"[{server_name}] Custom field '{ref}' not found in schema definitions.")
            logger.debug(f"[{server_name}] Original reference: {original_ref}")
            logger.debug(
                f"[{server_name}] Available definitions: {list(schema_defs.keys()) if schema_defs else 'None'}")

            # try to find a match in schema_defs
            if schema_defs:
                possible_matches = [
                    key for key in schema_defs.keys()
                    if ref.lower() in key.lower() or key.lower() in ref.lower()
                ]
                if possible_matches:
                    logger.info(f"[{server_name}] Found possible matches for '{ref}': {possible_matches}")
                    ref = possible_matches[0]
                    logger.info(f"[{server_name}] Using '{ref}' as fallback for '{original_ref}'")
                else:
                    # if no matches found, use Any type
                    logger.warning(f"[{server_name}] No matches found for '{ref}', using Any type")
                    default_value = ... if is_required else None
                    return Any, Field(
                        default=default_value,
                        description=f"Reference '{original_ref}' not found in schema definitions"
                    )
            else:
                # schema_defs is None, use Any type
                logger.warning(
                    f"[{server_name}] No schema definitions available for reference '{original_ref}', using Any type")
                default_value = ... if is_required else None
                return Any, Field(
                    default=default_value,
                    description=f"Reference '{original_ref}' cannot be resolved (no schema definitions)"
                )

        # 如果找到了引用，继续处理
        prop_schema = schema_defs[ref]

    prop_type = prop_schema.get("type")
    prop_desc = prop_schema.get("description", "")

    default_value = ... if is_required else prop_schema.get("default", None)
    pydantic_field = Field(default=default_value, description=prop_desc)

    # Handle the case where prop_type is missing but 'anyOf' key exists
    # In this case, use data type from 'anyOf' to determine the type hint
    if "anyOf" in prop_schema:
        type_hints = []
        for i, schema_option in enumerate(prop_schema["anyOf"]):
            try:
                type_hint, _ = _process_schema_property(
                    _model_cache,
                    schema_option,
                    f"{model_name_prefix}_{prop_name}",
                    f"choice_{i}",
                    False,
                    schema_defs,
                    server_name,
                )
                type_hints.append(type_hint)
            except Exception as e:
                logger.warning(f"[{server_name}] Failed to process anyOf option {i} for {prop_name}: {e}")
                type_hints.append(Any)  # 添加 Any 作为 fallback

        if type_hints:
            return Union[tuple(type_hints)], pydantic_field
        else:
            return Any, pydantic_field

    # Handle the case where prop_type is a list of types, e.g. ['string', 'number']
    if isinstance(prop_type, list):
        # Create a Union of all the types
        type_hints = []
        for type_option in prop_type:
            try:
                # Create a temporary schema with the single type and process it
                temp_schema = dict(prop_schema)
                temp_schema["type"] = type_option
                type_hint, _ = _process_schema_property(
                    _model_cache, temp_schema, model_name_prefix, prop_name, False, schema_defs, server_name
                )
                type_hints.append(type_hint)
            except Exception as e:
                logger.warning(f"[{server_name}] Failed to process type option {type_option} for {prop_name}: {e}")
                type_hints.append(Any)

        # Return a Union of all possible types
        if type_hints:
            return Union[tuple(type_hints)], pydantic_field
        else:
            return Any, pydantic_field

    if prop_type == "object":
        nested_properties = prop_schema.get("properties", {})
        nested_required = prop_schema.get("required", [])
        nested_fields = {}

        nested_model_name = f"{model_name_prefix}_{prop_name}_model".replace(
            "__", "_"
        ).rstrip("_")

        if nested_model_name in _model_cache:
            return _model_cache[nested_model_name], pydantic_field

        for name, schema in nested_properties.items():
            try:
                is_nested_required = name in nested_required
                nested_type_hint, nested_pydantic_field = _process_schema_property(
                    _model_cache,
                    schema,
                    nested_model_name,
                    name,
                    is_nested_required,
                    schema_defs,
                    server_name,
                )
                if name_needs_alias(name):
                    other_names = set().union(nested_properties, nested_fields, _model_cache)
                    alias_name = generate_alias_name(name, other_names)
                    aliased_field = Field(
                        default=nested_pydantic_field.default,
                        description=nested_pydantic_field.description,
                        alias=name
                    )
                    nested_fields[alias_name] = (nested_type_hint, aliased_field)
                else:
                    nested_fields[name] = (nested_type_hint, nested_pydantic_field)
            except Exception as e:
                logger.warning(f"[{server_name}] Failed to process nested property {name} in {nested_model_name}: {e}")
                # use any as fallback
                is_nested_required = name in nested_required
                default_value = ... if is_nested_required else None
                nested_fields[name] = (Any, Field(default=default_value, description=f"Failed to process: {str(e)}"))

        if not nested_fields:
            return Dict[str, Any], pydantic_field

        try:
            NestedModel = create_model(nested_model_name, **nested_fields)
            _model_cache[nested_model_name] = NestedModel
            return NestedModel, pydantic_field
        except Exception as e:
            logger.error(f"[{server_name}] Failed to create nested model {nested_model_name}: {e}")
            return Dict[str, Any], pydantic_field

    elif prop_type == "array":
        items_schema = prop_schema.get("items")
        if not items_schema:
            # Default to list of anything if items schema is missing
            return List[Any], pydantic_field

        try:
            # Recursively determine the type of items in the array
            item_type_hint, _ = _process_schema_property(
                _model_cache,
                items_schema,
                f"{model_name_prefix}_{prop_name}",
                "item",
                False,  # Items aren't required at this level,
                schema_defs,
                server_name,
            )
            list_type_hint = List[item_type_hint]
            return list_type_hint, pydantic_field
        except Exception as e:
            logger.warning(f"[{server_name}] Failed to process array items for {prop_name}: {e}")
            return List[Any], pydantic_field

    elif prop_type == "string":
        return str, pydantic_field
    elif prop_type == "integer":
        return int, pydantic_field
    elif prop_type == "boolean":
        return bool, pydantic_field
    elif prop_type == "number":
        return float, pydantic_field
    elif prop_type == "null":
        return None, pydantic_field
    else:
        # 处理未知类型
        if prop_type is not None:
            logger.warning(f"[{server_name}] Unknown property type '{prop_type}' for {prop_name}, using Any")
        return Any, pydantic_field


def get_model_fields(form_model_name, properties, required_fields, schema_defs=None, server_name="unknown"):
    model_fields = {}
    _model_cache: Dict[str, Type] = {}

    logger.debug(f"[{server_name}] Processing model '{form_model_name}' with {len(properties)} properties")
    logger.debug(f"[{server_name}] Available schema definitions: {list(schema_defs.keys()) if schema_defs else 'None'}")

    for param_name, param_schema in properties.items():
        try:
            if "$ref" in param_schema:
                logger.debug(f"[{server_name}] Processing property '{param_name}' with $ref: {param_schema['$ref']}")

            is_required = param_name in required_fields
            python_type_hint, pydantic_field_info = _process_schema_property(
                _model_cache,
                param_schema,
                form_model_name,
                param_name,
                is_required,
                schema_defs,
                server_name,
            )
            # Use the generated type hint and Field info
            if name_needs_alias(param_name):
                other_names = set().union(properties, model_fields, _model_cache)
                alias_name = generate_alias_name(param_name, other_names)
                aliased_field = Field(
                    default=pydantic_field_info.default,
                    description=pydantic_field_info.description,
                    alias=param_name
                )
                # Use the generated type hint and Field info
                model_fields[alias_name] = (python_type_hint, aliased_field)
            else:
                model_fields[param_name] = (python_type_hint, pydantic_field_info)

            logger.debug(f"[{server_name}] Successfully processed property '{param_name}' with type {python_type_hint}")

        except Exception as e:
            logger.error(f"[{server_name}] Failed to process property '{param_name}' in model '{form_model_name}': {e}")
            logger.error(f"[{server_name}] Property schema: {param_schema}")

            is_required = param_name in required_fields
            default_value = ... if is_required else None
            model_fields[param_name] = (
                Any,
                Field(
                    default=default_value,
                    description=f"Failed to process schema: {str(e)}"
                )
            )
            logger.warning(f"[{server_name}] Using fallback type 'Any' for property '{param_name}'")

    return model_fields


def get_tool_handler(
        session,
        endpoint_name,
        form_model_fields,
        response_model_fields=None,
):
    worker_id = os.getpid()

    if form_model_fields:
        FormModel = create_model(f"{endpoint_name}_form_model", **form_model_fields)

        # 总是尝试创建响应模型
        ResponseModel = Any
        if response_model_fields:
            try:
                ResponseModel = create_model(f"{endpoint_name}_response_model", **response_model_fields)
            except Exception as e:
                logger.warning(f"Failed to create response model for {endpoint_name}: {e}")

        def make_endpoint_func(endpoint_name: str, FormModel, session: ClientSession):
            async def tool(form_data: FormModel) -> ResponseModel:
                args = form_data.model_dump(exclude_none=True, by_alias=True)
                print(f"[Worker {worker_id}] Calling endpoint: {endpoint_name}, with args: {args}")

                try:
                    result = await session.call_tool(endpoint_name, arguments=args)

                    if result.isError:
                        error_message = "Unknown tool execution error"
                        error_data = None
                        if result.content:
                            if isinstance(result.content[0], types.TextContent):
                                error_message = result.content[0].text
                        detail = {"message": error_message}
                        if error_data is not None:
                            detail["data"] = error_data
                        raise HTTPException(
                            status_code=500,
                            detail=detail,
                        )

                    response_data = process_tool_response(result)
                    final_response = (
                        response_data[0] if len(response_data) == 1 else response_data
                    )

                    # 智能响应适配：尝试让 Pydantic 验证，失败则自动转换
                    final_response = smart_response_adapter(final_response, ResponseModel, endpoint_name)

                    return final_response

                except McpError as e:
                    print(
                        f"[Worker {worker_id}] MCP Error calling {endpoint_name}: {traceback.format_exc()}"
                    )
                    status_code = MCP_ERROR_TO_HTTP_STATUS.get(e.error.code, 500)
                    raise HTTPException(
                        status_code=status_code,
                        detail=(
                            {"message": e.error.message, "data": e.error.data}
                            if e.error.data is not None
                            else {"message": e.error.message}
                        ),
                    )
                except Exception as e:
                    print(
                        f"[Worker {worker_id}] Unexpected error calling {endpoint_name}: {traceback.format_exc()}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail={"message": "Unexpected error", "error": str(e)},
                    )

            return tool

        tool_handler = make_endpoint_func(endpoint_name, FormModel, session)
    else:

        def make_endpoint_func_no_args(
                endpoint_name: str, session: ClientSession
        ):  # Parameterless endpoint
            async def tool():  # No parameters
                print(
                    f"[Worker {worker_id}] Calling endpoint: {endpoint_name}, with no args"
                )
                try:
                    result = await session.call_tool(
                        endpoint_name, arguments={}
                    )  # Empty dict

                    if result.isError:
                        error_message = "Unknown tool execution error"
                        if result.content:
                            if isinstance(result.content[0], types.TextContent):
                                error_message = result.content[0].text
                        detail = {"message": error_message}
                        raise HTTPException(
                            status_code=500,
                            detail=detail,
                        )

                    response_data = process_tool_response(result)
                    final_response = (
                        response_data[0] if len(response_data) == 1 else response_data
                    )

                    # 对无参数端点也进行智能响应适配
                    final_response = smart_response_adapter(final_response, Any, endpoint_name)

                    return final_response

                except McpError as e:
                    print(
                        f"[Worker {worker_id}] MCP Error calling {endpoint_name}: {traceback.format_exc()}"
                    )
                    status_code = MCP_ERROR_TO_HTTP_STATUS.get(e.error.code, 500)
                    # Propagate the error received from MCP as an HTTP exception
                    raise HTTPException(
                        status_code=status_code,
                        detail=(
                            {"message": e.error.message, "data": e.error.data}
                            if e.error.data is not None
                            else {"message": e.error.message}
                        ),
                    )
                except Exception as e:
                    print(
                        f"[Worker {worker_id}] Unexpected error calling {endpoint_name}: {traceback.format_exc()}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail={"message": "Unexpected error", "error": str(e)},
                    )

            return tool

        tool_handler = make_endpoint_func_no_args(endpoint_name, session)

    return tool_handler


def _convert_dict_list_to_str_list(dict_list: List[dict]) -> List[str]:
    """将字典列表转换为字符串列表"""
    result = []
    for item in dict_list:
        if isinstance(item, dict):
            result.append(extract_string_from_dict(item))
        else:
            result.append(str(item))
    return result


def _convert_str_list_to_int_list(str_list: List[str]) -> List[int]:
    """将字符串列表转换为整数列表"""
    result = []
    for item in str_list:
        try:
            result.append(int(item))
        except (ValueError, TypeError):
            # 尝试从字符串中提取数字
            import re
            numbers = re.findall(r'\d+', str(item))
            result.append(int(numbers[0]) if numbers else 0)
    return result


def _convert_str_list_to_float_list(str_list: List[str]) -> List[float]:
    """将字符串列表转换为浮点数列表"""
    result = []
    for item in str_list:
        try:
            result.append(float(item))
        except (ValueError, TypeError):
            # 尝试从字符串中提取数字
            import re
            numbers = re.findall(r'\d+\.?\d*', str(item))
            result.append(float(numbers[0]) if numbers else 0.0)
    return result