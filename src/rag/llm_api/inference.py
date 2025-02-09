import os
import time
import json
import logging
from pydantic import BaseModel
from typing import Type, List, Dict
import tiktoken
import openai
import anthropic
from openai import OpenAI
from .llm_registry import (
    get_registered_models, 
    get_token_encoding_model,
)

# 全局 token 计数器，不能用于多线程。
TOKENS_IN = dict()
TOKENS_OUT = dict()
LLM_COST = dict()    # 某一个 LLM 的累计费用

logger = logging.getLogger(__name__)

def curr_cost_est(prompt=None, 
                  system_prompt=None, 
                  answer=None, 
                  model_key=None, 
                  usage=None, 
                  print_cost=True,
                  llm_config=None) -> float:
    """
    计算当前消耗的费用（近似值）。
    
    Args:
        prompt (str, optional): 用户输入的提示文本
        system_prompt (str, optional): 系统提示文本
        answer (str, optional): 模型返回的答案文本
        model_key (str, optional): 模型标识符
        print_cost (bool, optional): 是否打印费用信息
    
    Returns:
        float: 预估的总费用（美元）
    """
    try:
        if model_key not in TOKENS_IN:
            TOKENS_IN[model_key] = 0.0
            TOKENS_OUT[model_key] = 0.0
            
        if model_key not in LLM_COST:
            LLM_COST[model_key] = 0.0
        
        if usage:
            usage = usage.to_dict()
            input_tokens = usage.get("prompt_tokens", 0)    # for qwen llm 
            output_tokens = usage.get("completion_tokens", 0)
            TOKENS_IN[model_key] += input_tokens
            TOKENS_OUT[model_key] += output_tokens
            
        else:
            if all(x is not None for x in [prompt, system_prompt, answer, model_key]):
                try:
                    encoding_model = get_token_encoding_model(model_key)
                    encoding = tiktoken.encoding_for_model(encoding_model)
                    TOKENS_IN[model_key] += len(encoding.encode(system_prompt + prompt))
                    TOKENS_OUT[model_key] += len(encoding.encode(answer))
                except Exception as e:
                    if print_cost:
                        logger.warning(f"Token计算出现错误: {e}")
            
        # 计算总费用
        if llm_config["costs"]:
            cost_config = {}
            cost_config["input_costs"] = llm_config["costs"]["input_costs"]
            cost_config["output_costs"] = llm_config["costs"]["output_costs"]

            costmap_in = llm_config["costs"]["input_costs"]/1000000
            costmap_out = llm_config["costs"]["output_costs"]/1000000
            
            LLM_COST[model_key] = costmap_in * TOKENS_IN[model_key] + costmap_out * TOKENS_OUT[model_key]
            total_cost = LLM_COST[model_key]
            
            if print_cost:
                if llm_config["costs"]["cost_currency"] == "USD":
                    logger.info(f"当前累计消耗: ${total_cost:.4f} (预估值，可能与实际费用有偏差)")
                elif llm_config["costs"]["cost_currency"] == "CNY":
                    logger.info(f"当前累计消耗: ￥{total_cost:.4f} (预估值，可能与实际费用有偏差)")
                else:
                    logger.warning(f"未配置 API 费用 currency，请检查 llm_config.yaml 文件。")
        else:
            logger.warning("未配置费用信息，无法计算费用。请检查 llm_config.yaml 文件。")
        
        return total_cost
    
    except Exception as e:
        if print_cost:
            logger.error(f"费用计算出现错误: {e}")
        return 0.0

class QueryLLMs:
    """
    QueryLLMs 封装了调用多种大语言模型的查询逻辑，
    并支持从环境变量和传入参数中获取 API key，模型配置也全部提取到固定的字典中。
    """
    
    def __init__(self, llm_config=None):
        """
        初始化 QueryLLMs 实例。
        
        参数：
            api_config (str): OpenAI 的 API key。
            anthropic_api_key (str): Anthropic 的 API key。
            deepseek_api_key (str): DeepSeek 的 API key。
            version (str): 版本号，用于区分不同调用方式（例如 "0.28" 为旧版本）。
        """
        self.api_config = llm_config
        self.api_keys = self._get_api_keys(llm_config)
        self.model_name = llm_config["model_name"]
        self.MODEL_CONFIG = get_registered_models()    # TODO del
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def _get_api_keys(api_config=None):
        """
        从参数或环境变量中获取并验证 API key，并设置到环境变量中。
        
        首先从配置中获取，如果没有则从环境变量中获取。
        
        若未能获取到任何一个 key，则抛出异常。
        """
        if api_config['supplier'] == "qwen":
            if api_config['api_key'] and api_config['api_key'] != "YOUR_API_KEY":
                logger.info("使用配置中的 API key")
            else:
                logger.info("环境变量中的 API key，已配置到参数中")
                api_config['api_key'] = os.getenv('DASHSCOPE_API_KEY')
        
        if api_config['supplier'] == "openai":
            if api_config['api_key'] and api_config['api_key'] != "YOUR_API_KEY":
                logger.info("使用配置中的 API key")
            else:
                logger.info("使用环境变量中的 API key")
                api_config['api_key'] = os.getenv('OPENAI_API_KEY')
                
    def format_structured_output_illustration_structure(self, format_json_schema) -> tuple:
        """
        迭代 format_json_schema 中的所有元素，按照指定格式输出字符串和 JSON。

        Args:
            format_json_schema (dict): 定义响应格式的 JSON schema。
                示例:
                {
                    "title": "DetermineRoleSchema",
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "title": "File Name",
                            "description": "选定角色信息所在的作品"
                        },
                        "name": {
                            "type": "string",
                            "title": "Name",
                            "description": "选定角色名字"
                        }
                    },
                    "required": ["file_name", "name"]
                }

        Returns:
            tuple: 包含以下三个元素的元组：
                - json_illustration (str): 按指定格式整理的字符串说明
                    示例:
                    key 标题: "file_name"
                    value 描述: "选定角色信息所在的作品"
                    value 类型: "string"

                    key 标题: "name"
                    value 描述: "选定角色名字"
                    value 类型: "string"

                - json_structure_dict (dict): 字段描述的字典结构
                    示例: {'file_name': '选定角色信息所在的作品', 'name': '选定角色名字'}

                - response_format_json_schema_client (dict): 从 response_format_json_schema 重构的客户端使用的 JSON schema
                    示例:
                    {
                        "type": "json_object",
                        "properties": {
                            "file_name": {
                                "type": "string",
                                "description": "选定角色信息所在的作品"
                            },
                            "name": {
                                "type": "string",
                                "description": "选定角色名字"
                            }
                        }
                    }
        """
        properties = format_json_schema.get('properties', {})
        json_illustration = ""
        json_structure_dict = {}
        response_format_json_schema_client = {
            "type": "json_object",
            "properties": {}
        }
        
        for key, value in properties.items():
            json_illustration += f"""\
    key 标题: "{key}"
    value 描述: "{value.get('description', '')}"
    value 类型: "{value.get('type', '')}"
    
    """
            json_structure_dict[key] = value.get('description', '')
            response_format_json_schema_client["properties"][key] = {
                "type": value.get("type", ""),
                "description": value.get("description", "")
            }
        
        return json_illustration.strip(), json_structure_dict, response_format_json_schema_client
    
    def format_structured_output(self, messages, response_format):
        """
        qwen 等请求需要手动添加 json 输出要求到 prompt 中。
        将 pydantic 的 response_format 转换为 json 输出要求，添加至 Prompt 末尾。
        将 pydantic 的 response_format 转换为 json 格式，作为 response_format 的参数。
        https://help.aliyun.com/zh/model-studio/user-guide/json-mode?spm=a2c4g.11186623.help-menu-search-2400256.d_0
        """
        response_format_json_schema = response_format.model_json_schema()
        response_format_structured_output_illustration, response_format_json_structure, response_format_json_schema_client = self.format_structured_output_illustration_structure(response_format_json_schema)
        
        messages[0]["content"] += f"\n\n你需要提取出给定字段内容：\n\n 字段说明:\n{response_format_structured_output_illustration}\n请输出 JSON 字符串，不要输出其它无关内容。\n示例：{response_format_json_structure}"    # system prompt
        messages[1]["content"] += f"\n\n请按照以下 JSON 格式输出结果：\n{response_format_json_structure}"    # user prompt
        
        return messages, response_format_json_schema_client
    
    def _call_openai(self, messages: List[Dict], 
                     temperature: float = 0.7, 
                     response_format: Type[BaseModel] = None,
                     ):
        """
        调用 OpenAI 类模型（包括 DeepSeek 等基于 OpenAI 接口的模型）。
        """
        # format prompt
        if self.api_config['supplier'] == "qwen" and response_format:
            messages, response_format = self.format_structured_output(messages, response_format)
        
        # request llms client
        client = OpenAI(
            api_key=self.api_config['api_key'], 
            base_url=self.api_config.get("base_url"))
        params = {
            "model": self.model_name, 
            "messages": messages,
            "temperature": temperature
            }
        if response_format:
            params["response_format"] = response_format
        
        if self.api_config['supplier'] == "openai" and response_format:
            completion = client.beta.chat.completions.parse(**params)    # for openai structured_output https://platform.openai.com/docs/guides/structured-outputs#tips-for-your-data-structure
        else:
            completion = client.chat.completions.create(**params)
        
        
        if completion.usage:
            return completion.choices[0].message.content, completion.usage
        else:
            return completion.choices[0].message.content, None
    
    def run(self, 
            prompt: str, 
            system_prompt: str = 'You are a helpful assistant.', 
            response_format: Type[BaseModel] = None,
            tries: int = 5, 
            timeout: float = 5.0, 
            temperature: float = 0.7, 
            print_cost: bool = True):
        """
        先根据请求类型整理请求参数，重试指定次数直至成功返回结果。
        
        参数：
            model_key (str): 用于查询的模型标识符。
            prompt (str): 用户输入的提示。
            system_prompt (str): 系统提示。 默认值为 'You are a helpful assistant.'
            tries (int): 最大重试次数。
            timeout (float): 重试间隔秒数。
            temperature (float): 生成温度参数。
            print_cost (bool): 是否打印当前消耗费用。
        
        返回：
            str: 模型返回的文本答案。
        """

            
        # call llm client
        attempt = 0
        usage = None
        while attempt < tries:
            try:
                if self.api_config["client"] == "openai":
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    answer, usage = self._call_openai(
                        messages=messages, 
                        temperature=temperature, 
                        response_format=response_format,
                        )
                
                    curr_cost_est(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        answer=answer,
                        model_key=self.model_name,
                        usage=usage,
                        print_cost=print_cost,
                        llm_config=self.api_config
                    )
                    
                    return answer
                else:
                    raise ValueError(f"模型 '{self.model_name}' 不受支持。")
                    

            except Exception as e:
                attempt += 1
                if attempt == tries:
                    raise e
                self.logger.warning(f"尝试 {attempt}/{tries} 失败: {str(e)}")
                time.sleep(timeout)
        
        raise Exception(f"在 {tries} 次尝试后仍未成功")