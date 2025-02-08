from typing import Dict, Optional
from llm_api.inference import QueryModel

# Define the base prompt as a constant
REWRITE_QUERY_PROMPT = (
    "拓展原始查询以更好地补充信息，说明在什么样的情形下，什么人在什么地方做了什么事情。\n"
    "重写时请考虑以下几点：\n"
    "1. 保留查询的核心语义\n"
    "2. 添加更多角色相关的描述词，比如外貌、语言、动作、心理、神态。\n"
    "3. 根据原始查询的信息，增加合适的信息，新增的人物、地点、情形需符合原始查询的逻辑\n"
    "\n"
    "原始查询：{query}\n"
)

class QueryRewritingAgent:
    def __init__(self, llm_config: QueryModel):
        """
        初始化查询重写类
        Args:
            llm_config: 
        """
        self.llm_api = QueryModel(llm_config)
        
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
        rewritten_query = self.llm_api.run(
            prompt,
            temperature=0.3  # 使用较低的temperature以获得更确定性的结果
        )
        return rewritten_query

    def _build_rewrite_prompt(self, query: str, 
                           context: Optional[Dict]) -> str:
        """
        构建重写提示
        Args:
            run: 原始查询
            context: 上下文信息
        Returns:
            构建的提示词
        """
        # Use the constant and format it with the query
        base_prompt = REWRITE_QUERY_PROMPT.format(query=query)
                
        return base_prompt 