import os
import yaml
import logging
from dotenv import load_dotenv
from typing import Dict
import datetime

from llm_api.inference import AgencyLLMs
from llm_api.query_rewriting import QueryRewritingAgent
from agent.role_matching_agent import RoleMatching
from index_builder.index_builder import IndexBuilder
from rag_service.service import RAGService

# 加载环境变量
load_dotenv()

def setup_logging():
    """设置日志"""
    # 确保日志目录存在
    os.makedirs('log', exist_ok=True)
    
    # 生成日志文件名（使用时间戳）
    log_filename = f"log/rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

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
    model_name = "qwen-max-latest"    # "gpt-4o-mini"

    try:
        # 加载配置
        config = load_config("config/config.yaml")
        llm_config = load_config("config/llm_config.yaml")
        
        # 设置 HF 代理
        if config["general"]["HF_ENDPOINT"]:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 初始化各个组件
        query_rewriting = QueryRewritingAgent(
            llm_config=llm_config[model_name]
        )
        index_builder = IndexBuilder(config)
        vectorstore = index_builder.load_index(config["vectorstore"]["persist_directory"]+'/'+config["vectorstore"]["metadata_path"].split('/')[-1]+'_'+config["vectorstore"]["embedding_model"].split('/')[-1])   # eg. data/vectorstore/retrieval_traits_all-MiniLM-L6-v2
        role_matching = RoleMatching(
            config=config,
            llm_config=llm_config[model_name],
            vectordb=vectorstore            
        )


        # 示例查询
        original_query = "蜂鸟类中，蜂鸟独有一对椭圆形骨，即籽骨，嵌入在扩张的十字韧带腱膜尾部。这块籽骨支撑着多少对肌腱？请用数字回答。"
        
        # 1. 重写查询
        logger.info("重写查询...")
        rewritten_query = query_rewriting.rewrite_query_agent(original_query=original_query)
        logger.info(f"重写后的查询: {rewritten_query}")

        # 2. 确定角色
        logger.info("确定角色...")
        
        # 使用新的函数名和参数
        role_info = role_matching.determine_roles_agent(
            query=original_query,
            rewritten_query=rewritten_query,
        )

        test = "test"
        # 任务1：完成单轮对话         

        # 输入 

        # 1. 当前角色
        # 2. 当前角色所在的文学作品
        # 3. 用户输入
        # 4. 环境感知

        # 输出

        # answer

        # 任务2：完成多轮对话      

        # 输入 

        # 1. 当前角色
        # 2. 当前角色所在的文学作品
        # 3. 用户输入
        # 4. 环境感知
        # 5. 历史对话

        # 输出

        # answer

        # 任务3：faction call 将 CoT 模式嵌入到任务1 与 任务2 中。

        # 输入：

        # 1. 上下文
        # 2. 判断是否开启 CoT 模式
        # 3. 之前所有的 CoT 内容

        # 输出：

        # bool
        
    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 