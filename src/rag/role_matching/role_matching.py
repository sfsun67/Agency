from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import logging

class RoleMatching:
    def __init__(self, config: Dict):
        """
        初始化角色确定类
        Args:
            config: 配置字典，包含向量存储路径等信息
            top_k: 返回的相似度最高的k个角色
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.top_k = config["role_matching"]["top_k"]
        self.similarity_threshold = config["role_matching"]["similarity_threshold"]
        self.main_character_threshold = config["role_matching"]["main_character_threshold"]

    def determine_roles(self, run: str) -> List[Dict]:
        """
        确定角色
        Args:
            run: 查询文本
        Returns:
            角色信息列表
        """
        try:
            results = self.vectorstore.similarity_search_with_scores(
                run, k=self.top_k
            )
            
            roles = [self._format_role_info(doc, score) 
                    for doc, score in results 
                    if score >= self.similarity_threshold]
            
            # 计算主角信息
            if roles:
                self._calculate_main_character_info(roles)
                
            return roles
        except Exception as e:
            self.logger.error(f"角色确定失败: {str(e)}")
            raise

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