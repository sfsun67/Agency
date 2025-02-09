import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import logging
from utils.prompt import BasePrompt
from llm_api.inference import AgencyLLMs
from rag_service.service import RAGService

class DetermineRoleSchema(BaseModel):
    file_name: str = Field(..., description="选定角色信息所在的作品")
    name: str = Field(description="选定角色名字")

class DetermineRolePrompt(BasePrompt):
    doc_list_str: str
    query: str
    prompt_template: str = """请你分析，在下面的文学作品中，那个角色的背景和自我认知能够最好的回答下面的问题？
候选角色信息：
{self.doc_list_str}

这个角色需要分析的问题：
{self.query}

请给出你的分析，并输出这个人的名字：
"""
    # 新增一个函数，用于格式化 doc_list
    @staticmethod
    def format_doc_list(doc_list: Dict[str, Dict[str, str]]) -> str:
        """
        将 doc_list 格式化为指定输出形式的字符串：
            作品名称 filename：
            人格特质 personalities_trails：
            自我认知 self_awareness：
        """
        output_lines = []
        for filename, info in doc_list.items():
            personalities = info.get("personalities_trails", "N/A")
            self_awareness = info.get("self_awareness", "N/A")
            # 拼接格式：作品名称、人格特质、自我认知
            output_lines.append(
                f"作品名称: {filename}\n"
                f"人格特质: {personalities}\n"
                f"自我认知: {self_awareness}\n"
            )
        return "\n".join(output_lines)



class RoleMatching:
    def __init__(self, 
                 config: Dict,
                 llm_config: Dict,
                 vectordb: Chroma
                 ):
        """
        初始化角色确定类
        Args:
            config: 配置字典，包含向量存储路径等信息
            top_k: 返回的相似度最高的k个角色
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.llm_api = AgencyLLMs(llm_config)
        self.vectordb = vectordb
        
        self.top_k = config["role_matching"]["top_k"]
        self.main_character_threshold = config["role_matching"]["main_character_threshold"]


    
    def find_most_frequent_substring(self, string_list: List[str], target_substring: str) -> str:
        """
        找到字符串列表中，出现次数最多的子字符串
        Args:
            string_list: 字符串列表
            target_substring: 目标子字符串
        Returns:
            出现次数最多的子字符串
        """
        return max(string_list, key=lambda s: s.count(target_substring))

    def determine_roles_agent(
        self, 
        query: str,
        rewritten_query: str,) -> List[Dict]:
        """
        确定角色, 问模型，在检索相似的文本中，谁的背景和自我认知能够最好的回答 "" 问题？
        Args:
            query: 查询文本
            rewritten_query: 重写后的查询文本
            vectordb: Chroma向量数据库实例
        Returns:
            角色信息列表
        """
        # 调用检索相似文本函数
        rag = RAGService(
            config=self.config,
            vectordb=self.vectordb,
            llm_api=self.llm_api
        )
        retriever_docs = rag.retrieve_similar_texts(rewritten_query)
        
        doc_list = {}
        for doc, _ in retriever_docs:
            filename = doc.metadata['file_name']
            personalities_trails = doc.metadata['output-personalities_trails']
            self_awareness = doc.metadata['output-self_awareness']
            doc_list[filename] = {
                "personalities_trails": personalities_trails,
                "self_awareness": self_awareness
            }
        
        # 调整 prompt，请求 llms
        doc_list_str = DetermineRolePrompt.format_doc_list(doc_list)
        determine_role_prompt = DetermineRolePrompt(
                doc_list_str=doc_list_str,
                query=query
            ).render_prompt()
        
        determine_role = self.llm_api.run(
            prompt=determine_role_prompt,
            temperature=0.7,
            response_format=DetermineRoleSchema
            )

        # 返回字段验证
        if isinstance(determine_role, str):
            try:
                determine_role = json.loads(determine_role)
                determine_role = DetermineRoleSchema(**determine_role)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 解析失败: {str(e)}")
                raise ValueError("determine_role 不是有效的 JSON 字符串")
            
            
        # 确保返回的文件名是正确的
        # 获取所有文件名列表
        available_filenames = list(doc_list.keys())
        
        # 使用 find_most_frequent_substring 找到最匹配的文件名
        matched_filename = self.find_most_frequent_substring(
            available_filenames, 
            determine_role.file_name
        )
        
        # 更新 determine_role 中的文件名为匹配到的文件名
        determine_role.file_name = matched_filename
        self.logger.info(f"选定角色: {determine_role.name}, 来源: {determine_role.file_name}")

        return determine_role

    # try:
    #     roles = [self._format_role_info(doc, score) 
    #             for doc, score in retriever_docs 
    #             if score >= self.similarity_threshold]
        
    #     # 计算主角信息
    #     if roles:
    #         self._calculate_main_character_info(roles)
            
    # except Exception as e:
    #     self.logger.error(f"角色确定失败: {str(e)}")
    #     raise
    
    # prompt = self.prompt_tpl.determine_roles_prompt.format(
    #     doc_list_str=doc_list_str,
    #     query=query
    # )

    def _format_role_info(self, doc, score: float) -> Dict:
        """
        格式化角色信息
        Args:
            doc: 文档对象
            score: 相似度分数
        Returns:
            格式化后的角色信息
        """
        return {
            "role_name": doc.metadata.get("role_name", "Unknown"),
            "source_files": doc.metadata.get("source_files", []),
            "source": doc.page_content,
            "lines": doc.metadata.get("lines", ""),
            "score": float(score),
            "is_main_character": False  # 默认不是主角，后续计算
        }

    def _calculate_main_character_info(self, roles: List[Dict]):
        """
        计算主角信息
        Args:
            roles: 角色信息列表
        """
        # 根据出现次数或其他指标计算是否是主角
        total_score = sum(role["score"] for role in roles)
        for role in roles:
            role["is_main_character"] = (
                role["score"] / total_score >= self.main_character_threshold
            ) 