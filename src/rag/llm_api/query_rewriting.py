from typing import Dict, Optional
from pydantic import BaseModel
from llm_api.inference import AgencyLLMs
from utils.prompt import BasePrompt

class RewriteQueryPrompt(BasePrompt):
    query: str
    prompt_template: str = """"你是一名教育心理学家，首先分析什么样的人能够正确回答下面的问题。然后根据问题说明这个人经历了什么样的事情，才能够回答正确这道问题。

这个人需要回答的问题是：
{self.query}

请考虑以下几点：
1. 保留问题的核心语义，不增加不确定的问题信息。
2. 添加更多角色相关的描述词，比如外貌、语言、动作、心理、神态。
3. 根据原始查询的信息，增加合适的信息，新增的人物、地点、情形需符合原始查询的逻辑。
4. 不要输出包含答案的信息。

你对这个人的描述是："""
    

class QueryRewritingAgent:
    def __init__(self, 
                 llm_config: AgencyLLMs, 
                 ):
        """
        初始化查询重写类
        Args:
            llm_config: 
        """
        self.llm_api = AgencyLLMs(llm_config)
        
    def rewrite_query_agent(self, original_query: str) -> str:
        """
        重写查询
        Args:
            original_query: 原始查询
            context: 上下文信息，可能包含角色信息、场景等
        Returns:
            重写后的查询
        """
        rewrite_query_prompt = RewriteQueryPrompt(query=original_query).render_prompt()
        rewritten_query = self.llm_api.run(
            prompt=rewrite_query_prompt,
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
        
        return rewrite_query_prompt 