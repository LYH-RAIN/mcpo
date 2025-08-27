"""
更优雅的响应适配器方案集合
提供多种兼容方式来处理 MCP 服务器返回数据与期望类型不匹配的问题
"""

import json
import logging
from typing import Any, Dict, List, Type, Union, get_origin, get_args
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


# 方案一：策略模式的适配器
class ResponseAdapter(ABC):
    """响应适配器基类"""
    
    @abstractmethod
    def can_adapt(self, response_data: Any, target_type: Type, endpoint_name: str) -> bool:
        """判断是否可以适配"""
        pass
    
    @abstractmethod
    def adapt(self, response_data: Any, target_type: Type, endpoint_name: str) -> Any:
        """执行适配"""
        pass


class ListDictToListStrAdapter(ResponseAdapter):
    """字典列表到字符串列表的适配器"""
    
    def can_adapt(self, response_data: Any, target_type: Type, endpoint_name: str) -> bool:
        origin = get_origin(target_type)
        args = get_args(target_type)
        return (origin is list and args and args[0] is str and 
                isinstance(response_data, list) and 
                response_data and isinstance(response_data[0], dict))
    
    def adapt(self, response_data: Any, target_type: Type, endpoint_name: str) -> Any:
        adapted_list = []
        for item in response_data:
            if isinstance(item, dict):
                adapted_list.append(self._extract_string_from_dict(item))
            else:
                adapted_list.append(str(item))
        
        logger.info(f"Adapted {endpoint_name}: List[dict] -> List[str] ({len(adapted_list)} items)")
        return adapted_list
    
    def _extract_string_from_dict(self, item_dict: dict) -> str:
        """从字典中提取字符串"""
        priority_fields = ['name', 'title', 'fullname', 'displayName', 'id', 'key', 'label']
        
        for field in priority_fields:
            if field in item_dict and isinstance(item_dict[field], str) and item_dict[field].strip():
                return item_dict[field]
        
        # 寻找第一个字符串值
        for key, value in item_dict.items():
            if isinstance(value, str) and value.strip():
                return value
        
        # 寻找数字
        for key, value in item_dict.items():
            if isinstance(value, (int, float)):
                return str(value)
        
        # 最后使用 JSON
        return json.dumps(item_dict, ensure_ascii=False, separators=(',', ':'))


class StrategyResponseAdapter:
    """策略模式的响应适配器管理器"""
    
    def __init__(self):
        self.adapters: List[ResponseAdapter] = [
            ListDictToListStrAdapter(),
            # 可以添加更多适配器
        ]
    
    def adapt(self, response_data: Any, target_type: Type, endpoint_name: str) -> Any:
        """尝试适配响应数据"""
        for adapter in self.adapters:
            if adapter.can_adapt(response_data, target_type, endpoint_name):
                return adapter.adapt(response_data, target_type, endpoint_name)
        
        return response_data  # 无法适配时返回原数据


# 方案二：配置化的适配规则
@dataclass
class AdaptationRule:
    """适配规则配置"""
    endpoint_pattern: str  # 端点名称模式（支持通配符）
    source_type: str      # 源数据类型描述
    target_type: str      # 目标类型描述
    adapter_func: str     # 适配函数名称


class ConfigurableResponseAdapter:
    """配置化的响应适配器"""
    
    def __init__(self, config_file: str = None):
        self.rules = self._load_default_rules()
        if config_file:
            self.rules.extend(self._load_rules_from_file(config_file))
    
    def _load_default_rules(self) -> List[AdaptationRule]:
        """加载默认适配规则"""
        return [
            AdaptationRule(
                endpoint_pattern="*_jobs",
                source_type="List[dict]",
                target_type="List[str]",
                adapter_func="dict_list_to_str_list"
            ),
            AdaptationRule(
                endpoint_pattern="list_*",
                source_type="List[dict]",
                target_type="List[str]",
                adapter_func="dict_list_to_str_list"
            ),
        ]
    
    def _load_rules_from_file(self, config_file: str) -> List[AdaptationRule]:
        """从配置文件加载规则"""
        # 实现配置文件加载逻辑
        return []
    
    def adapt(self, response_data: Any, target_type: Type, endpoint_name: str) -> Any:
        """根据配置规则适配响应数据"""
        import fnmatch
        
        for rule in self.rules:
            if fnmatch.fnmatch(endpoint_name, rule.endpoint_pattern):
                if self._matches_types(response_data, target_type, rule):
                    adapter_func = getattr(self, rule.adapter_func, None)
                    if adapter_func:
                        logger.info(f"Applying rule {rule.endpoint_pattern} to {endpoint_name}")
                        return adapter_func(response_data, target_type, endpoint_name)
        
        return response_data
    
    def _matches_types(self, response_data: Any, target_type: Type, rule: AdaptationRule) -> bool:
        """检查数据类型是否匹配规则"""
        # 简化的类型匹配逻辑
        if rule.source_type == "List[dict]" and rule.target_type == "List[str]":
            origin = get_origin(target_type)
            args = get_args(target_type)
            return (origin is list and args and args[0] is str and 
                    isinstance(response_data, list) and 
                    response_data and isinstance(response_data[0], dict))
        return False
    
    def dict_list_to_str_list(self, response_data: List[dict], target_type: Type, endpoint_name: str) -> List[str]:
        """字典列表转字符串列表"""
        result = []
        for item in response_data:
            if isinstance(item, dict):
                # 智能提取字符串
                priority_fields = ['name', 'title', 'fullname', 'displayName', 'id']
                extracted = None
                for field in priority_fields:
                    if field in item and isinstance(item[field], str):
                        extracted = item[field]
                        break
                
                if not extracted:
                    # 寻找第一个字符串值
                    for key, value in item.items():
                        if isinstance(value, str) and value.strip():
                            extracted = value
                            break
                
                result.append(extracted or str(item))
            else:
                result.append(str(item))
        
        return result


# 方案三：装饰器模式的适配器
def response_adapter(adapter_func=None, *, endpoint_patterns: List[str] = None):
    """响应适配装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # 获取端点名称（假设在某个位置可以获取到）
            endpoint_name = getattr(func, '__name__', 'unknown')
            
            if endpoint_patterns:
                import fnmatch
                if not any(fnmatch.fnmatch(endpoint_name, pattern) for pattern in endpoint_patterns):
                    return result
            
            if adapter_func:
                return adapter_func(result, endpoint_name)
            
            return result
        return wrapper
    return decorator


# 方案四：基于注解的自动适配
class AutoResponseAdapter:
    """基于类型注解的自动响应适配器"""
    
    @staticmethod
    def auto_adapt(response_data: Any, expected_type: Type, endpoint_name: str) -> Any:
        """基于类型注解自动适配"""
        if response_data is None:
            return response_data
        
        # 获取期望类型的信息
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        
        # 处理 List[T] 类型
        if origin is list and args:
            target_item_type = args[0]
            if isinstance(response_data, list) and response_data:
                first_item = response_data[0]
                
                # 如果期望 List[str] 但得到 List[dict]
                if target_item_type is str and isinstance(first_item, dict):
                    return AutoResponseAdapter._convert_dict_list_to_str_list(response_data)
                
                # 如果期望 List[int] 但得到 List[str]
                elif target_item_type is int and isinstance(first_item, str):
                    return AutoResponseAdapter._convert_str_list_to_int_list(response_data)
        
        # 处理单个值的转换
        elif expected_type is str and isinstance(response_data, dict):
            return AutoResponseAdapter._extract_string_from_dict(response_data)
        
        return response_data
    
    @staticmethod
    def _convert_dict_list_to_str_list(dict_list: List[dict]) -> List[str]:
        """将字典列表转换为字符串列表"""
        result = []
        for item in dict_list:
            if isinstance(item, dict):
                result.append(AutoResponseAdapter._extract_string_from_dict(item))
            else:
                result.append(str(item))
        return result
    
    @staticmethod
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
    
    @staticmethod
    def _extract_string_from_dict(item_dict: dict) -> str:
        """从字典中提取字符串"""
        priority_fields = ['name', 'title', 'fullname', 'displayName', 'id', 'key', 'label']
        
        for field in priority_fields:
            if field in item_dict and isinstance(item_dict[field], str) and item_dict[field].strip():
                return item_dict[field]
        
        for key, value in item_dict.items():
            if isinstance(value, str) and value.strip():
                return value
        
        for key, value in item_dict.items():
            if isinstance(value, (int, float)):
                return str(value)
        
        return json.dumps(item_dict, ensure_ascii=False, separators=(',', ':'))


# 使用示例和工厂函数
def create_response_adapter(strategy: str = "auto") -> Any:
    """创建响应适配器的工厂函数"""
    if strategy == "strategy":
        return StrategyResponseAdapter()
    elif strategy == "configurable":
        return ConfigurableResponseAdapter()
    elif strategy == "auto":
        return AutoResponseAdapter()
    else:
        raise ValueError(f"Unknown adapter strategy: {strategy}")