from typing import List, Dict, Optional
import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document
from llm_api.inference import QueryModel

from index_builder.index_builder import IndexBuilder

class RAGService:
    def __init__(self, config: Dict, llm_config: QueryModel, index_builder: IndexBuilder):
        """
        初始化RAG服务
        Args:
            config: 配置字典
            llm_api: LLMAPI实例
            index_builder: IndexBuilder实例
        """
        self.config = config
        self.llm_api = QueryModel(llm_config)
        self.index_builder = index_builder
        self.logger = logging.getLogger(__name__)

    def run(self, 
              question: str, 
              role_name: str,
              max_tokens: int = 2048) -> str:
        """
        查询并生成回答
        Args:
            question: 用户问题
            role_name: 角色名称
            max_tokens: 最大token数
        Returns:
            生成的回答
        """
        try:
            # 检索相关文本
            context_chunks = self.retrieve(question, role_name)
            if not context_chunks:
                return "抱歉，没有找到相关信息。"

            # 生成回答
            answer = self.generate_answer(
                context_chunks, 
                question,
                max_tokens=max_tokens
            )
            return answer

        except Exception as e:
            self.logger.error(f"查询失败: {str(e)}")
            raise

    def retrieve(self, 
                question: str, 
                role_name: str,
                top_k: int = 3) -> List[str]:
        """
        检索相关文本
        Args:
            question: 用户问题
            role_name: 角色名称
            top_k: 返回结果数量
        Returns:
            相关文本列表
        """
        try:
            # 加载角色索引
            vectorstore = self.index_builder.load_index(role_name)
            if vectorstore is None:
                self.logger.warning(f"角色 {role_name} 的索引不存在")
                return []

            # 执行检索
            results = vectorstore.similarity_search_with_scores(
                question, k=top_k
            )
            
            # 提取文本内容
            context_chunks = [doc.page_content for doc, _ in results]
            return context_chunks

        except Exception as e:
            self.logger.error(f"检索失败: {str(e)}")
            raise

    def generate_answer(self, 
                       context_chunks: List[str],
                       question: str,
                       max_tokens: int = 2048) -> str:
        """
        生成回答
        Args:
            context_chunks: 上下文文本列表
            question: 用户问题
            max_tokens: 最大token数
        Returns:
            生成的回答
        """
        try:
            # 构建提示词
            prompt = self._build_prompt(context_chunks, question)
            
            # 生成回答
            answer = self.llm_api.generate_text(
                prompt,
                temperature=self.config["rag"]["temperature"],
                max_tokens=max_tokens
            )
            return answer

        except Exception as e:
            self.logger.error(f"生成回答失败: {str(e)}")
            raise

    def _build_prompt(self, context_chunks: List[str], question: str) -> str:
        """
        构建提示词
        Args:
            context_chunks: 上下文文本列表
            question: 用户问题
        Returns:
            构建的提示词
        """
        context_text = "\n\n".join(context_chunks)
        prompt = (
            "基于以下上下文信息回答问题。如果无法从上下文中找到答案，"
            "请明确说明。请保持回答的准确性和相关性。\n\n"
            f"上下文信息：\n{context_text}\n\n"
            f"问题：{question}\n\n"
            "回答："
        )
        return prompt 