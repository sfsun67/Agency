import os
import yaml
import logging
from dotenv import load_dotenv
from typing import Dict

from llm_api.inference import QueryModel
from llm_api.query_rewriting import QueryRewritingAgent
from role_matching.role_matching import RoleMatching
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

def load_config(config_path: str) -> Dict:
    """加载配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def todo(human_mode, llm_backend):
    # TODO args
    ###################################################
    ##  Stages where human input will be requested  ###
    ###################################################
    human_in_loop = {
        "literature review":      human_mode,
        "plan formulation":       human_mode,
        "data preparation":       human_mode,
        "running experiments":    human_mode,
        "results interpretation": human_mode,
        "report writing":         human_mode,
        "report refinement":      human_mode,
    }

    ###################################################
    ###  LLM Backend used for the different phases  ###
    ###################################################
    agent_models = {
        "literature review":      llm_backend,
        "plan formulation":       llm_backend,
        "data preparation":       llm_backend,
        "running experiments":    llm_backend,
        "report writing":         llm_backend,
        "results interpretation": llm_backend,
        "paper refinement":       llm_backend,
    }
    
    # TODO 实现一个 Agent，用于扮演角色回答问题，执行 rag 等操作
    reviewers = ReviewersAgent(model=self.model_backbone, notes=self.notes, openai_api_key=self.openai_api_key)
    

def main():
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    model_name = "qwen-max-latest"

    try:
        # 加载配置
        config = load_config("config/config.yaml")
        llm_config = load_config("config/llm_config.yaml")
        
        # 设置 HF 代理
        if config["general"]["HF_ENDPOINT"]:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 初始化各个组件
        llm_api = QueryModel(
            llm_config=llm_config[model_name]
        )   # 所以可能不需要初始化这个
        query_rewriting = QueryRewritingAgent(
            llm_config=llm_config[model_name]
        )
        index_builder = IndexBuilder(config)
        vectorstore = index_builder.load_index(config["vectorstore"]["persist_directory"]+'/'+config["vectorstore"]["metadata_path"].split('/')[-1])    # # 读取大的向量库，如果没有，则新建一个
        role_matching = RoleMatching(config)
        rag_service = RAGService(
            config=config, 
            llm_config=llm_config[model_name],
            index_builder=index_builder,
        )

        # 示例查询
        original_query = "蜂鸟类中，蜂鸟独有一对椭圆形骨，即籽骨，嵌入在扩张的十字韧带腱膜尾部。这块籽骨支撑着多少对肌腱？请用数字回答。"
        
        # 1. 重写查询
        logger.info("重写查询...")
        rewritten_query = query_rewriting.rewrite_query_agent(
            original_query=original_query,
            context={"domain": "social class", "era": "mid-twentieth century"}   # TODO use context
        )
        logger.info(f"重写后的查询: {rewritten_query}")

        # 2. 确定角色
        logger.info("确定角色...")
        
        # for test 
        vectorstore.similarity_search_with_score(rewritten_query)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(rewritten_query)
        print(retrieved_docs)
        
        role = role_matching.determine_roles_agent(
            vectorstore, 
            rewritten_query
            )
        if not role:
            logger.warning("未找到相关角色")
            return

        # 2.1 选择最相关的角色，如果 o1 r1 这样的模型，给出了理由，那么我觉得可以不用这个做了。
        target_role = role[0]
        logger.info(f"选定角色: {target_role['role_name']}")
        
        # 2.2 为选的的角色建立数据库【optional】
        

        # 3. 使用RAG服务生成回答
        logger.info("生成回答...")
        answer = rag_service.run(
            rewritten_query,
            target_role["role_name"]
        )
        logger.info(f"生成的回答: {answer}")

    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 