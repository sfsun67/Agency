"""
LLM models registry configuration.
This module contains the configuration for all supported LLM models.
"""

import logging

logger = logging.getLogger(__name__)

def get_registered_models():
    """
    Returns a dictionary of registered LLM models with their configurations.
    
    Each model configuration contains:
    - client: The client type ("openai" or "anthropic")
    - model: The model identifier
    - base_url: (optional) Custom API endpoint
    """
    return {
        "gpt-4o-mini": {
            "client": "openai",
            "model": "gpt-4o-mini-2024-07-18"
        },
        "gpt-4o": {
            "client": "openai",
            "model": "gpt-4o-2024-08-06"
        },
        "claude-3.5-sonnet": {
            "client": "anthropic",
            "model": "claude-3-5-sonnet-latest"
        },
        "deepseek-chat": {
            "client": "openai",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1"
        },
        "o1-mini": {
            "client": "openai",
            "model": "o1-mini-2024-09-12"
        },
        "o1": {
            "client": "openai",
            "model": "o1-2024-12-17"
        },
        "o1-preview": {
            "client": "openai",
            "model": "o1-preview"
        },
        "qwen": {
            "client": "openai",
            "model": "qwen-2024-01-01"
        }
    }

def get_token_encoding_model(model_str: str) -> str:
    """
    Returns the appropriate encoding model name for token counting.
    """
    
    try:
        encoding_map = {
            "o1-preview": "gpt-4o",
            "o1-mini": "gpt-4o",
            "claude-3.5-sonnet": "gpt-4o",
            "o1": "gpt-4o",
            "deepseek-chat": "cl100k_base"
        }
    except:
        logger.warning(f"模型 {model_str} encoding map 不存在, 使用 'gpt-4o' 代替。更准确的 token 数量计算请修改函数 get_token_encoding_model 。")
        return "gpt-4o"
    return encoding_map.get(model_str, model_str)