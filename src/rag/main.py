import os
import yaml
import logging
from dotenv import load_dotenv
from typing import Dict

from llm_api.llm_api import LLMAPI
from llm_api.query_rewriting import QueryRewriting
from role_determination.role_determination import RoleDetermination
from index_builder.index_builder import IndexBuilder
from rag_service.rag_service import RAGService

# 加载环境变量
load_dotenv()

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """加载配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 加载配置
        config = load_config()
        
        # 设置 HF 代理
        if config["general"]["HF_ENDPOINT"]:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 初始化各个组件
        llm_api = LLMAPI(config["qwen"])
        query_rewriting = QueryRewriting(llm_api)
        index_builder = IndexBuilder(config)
        # bug mate 向量库存在，但依旧重新计算向量库了。
        vectorstore = index_builder.load_index(config["vectorstore"]["persist_directory"]+'/'+config["vectorstore"]["metadata_path"].split('/')[-1])    # # 读取大的向量库，如果没有，则新建一个
        role_determination = RoleDetermination(config)
        rag_service = RAGService(config, llm_api, index_builder)

        # 示例查询
        original_query = "The capitalist class of the mid-twentieth century were said to join the upper class because they:"
        
        # 1. 重写查询
        logger.info("重写查询...")
        rewritten_query = query_rewriting.rewrite_query(
            original_query,
            context={"domain": "social class", "era": "mid-twentieth century"}
        )
        logger.info(f"重写后的查询: {rewritten_query}")

        # 2. 确定角色
        logger.info("确定角色...")
        roles = role_determination.determine_roles(rewritten_query)
        if not roles:
            logger.warning("未找到相关角色")
            return

        # 选择最相关的角色
        target_role = roles[0]
        logger.info(f"选定角色: {target_role['role_name']}")

        # 3. 使用RAG服务生成回答
        logger.info("生成回答...")
        answer = rag_service.query(
            rewritten_query,
            target_role["role_name"]
        )
        logger.info(f"生成的回答: {answer}")

    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 