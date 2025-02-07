from typing import Dict, Optional
from .llm_api import LLMAPI

class QueryRewriting:
    def __init__(self, llm_api: LLMAPI):
        """
        初始化查询重写类
        Args:
            llm_api: LLMAPI实例
        """
        self.llm_api = llm_api

    def rewrite_query(self, original_query: str, 
                     context: Optional[Dict] = None) -> str:
        """
        重写查询
        Args:
            original_query: 原始查询
            context: 上下文信息，可能包含角色信息、场景等
        Returns:
            重写后的查询
        """
        prompt = self._build_rewrite_prompt(original_query, context)
        rewritten_query = self.llm_api.generate_text(
            prompt,
            temperature=0.3  # 使用较低的temperature以获得更确定性的结果
        )
        return rewritten_query

    def _build_rewrite_prompt(self, query: str, 
                           context: Optional[Dict]) -> str:
        """
        构建重写提示
        Args:
            query: 原始查询
            context: 上下文信息
        Returns:
            构建的提示词
        """
        base_prompt = (
            "请重写以下查询以更好地匹配角色检索需求。"
            "重写时请考虑以下几点：\n"
            "1. 保留查询的核心语义\n"
            "2. 添加更多角色相关的描述词\n"
            "3. 使用更精确的表达方式\n\n"
            f"原始查询：{query}\n"
        )
        
        if context:
            base_prompt += f"\n上下文信息：\n"
            for key, value in context.items():
                base_prompt += f"{key}: {value}\n"
                
        base_prompt += "\n重写后的查询："
        return base_prompt 