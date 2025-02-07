from typing import Dict, Optional, Any, List
import openai
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMAPI:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 LLMAPI 类
        Args:
            config: 配置字典，包含API密钥等信息
        """
        self.config = config
        self.setup_api_client()
        self.logger = logging.getLogger(__name__)

    def setup_api_client(self):
        """设置API客户端"""
        if self.config.get("api_type") == "azure":
            openai.api_type = "azure"
            openai.api_base = self.config["api_base"]
            openai.api_version = self.config["api_version"]
            openai.api_key = self.config["api_key"]
        else:
            openai.api_key = self.config["api_key"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_api(self, messages: List[Dict[str, str]], 
                temperature: float = 0.7,
                max_tokens: int = 2048) -> str:
        """
        调用API生成回复
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
        Returns:
            生成的回复文本
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.config["model_name"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"API调用失败: {str(e)}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        获取文本的向量表示
        Args:
            text: 输入文本
        Returns:
            文本的向量表示
        """
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"文本向量化失败: {str(e)}")
            raise

    def generate_text(self, prompt: str, 
                     temperature: float = 0.7,
                     max_tokens: int = 2048) -> str:
        """
        生成文本
        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大token数
        Returns:
            生成的文本
        """
        messages = [{"role": "user", "content": prompt}]
        return self.call_api(messages, temperature, max_tokens) 