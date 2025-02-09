# rag
```markdown
project/
├── config/
│   └── config.yaml           # 配置文件
├── llm_api/
│   ├── __init__.py
│   ├── llm_api.py           # LLMAPI 类
│   ├── query_rewriting.py   # QueryRewriting 类
│   └── utils.py             # 通用工具函数
├── role_determination/
│   ├── __init__.py
│   └── role_determination.py # RoleDetermination 类
├── index_builder/
│   ├── __init__.py
│   └── index_builder.py      # IndexBuilder 类
├── rag_service/
│   ├── __init__.py
│   └── rag_service.py        # RAGService 类
├── data/
│   └── vectorstore/          # 存放向量数据库文件
├── tests/
│   ├── __init__.py
│   ├── test_llm_api.py
│   ├── test_role_determination.py
│   ├── test_index_builder.py
│   └── test_rag_service.py
├── requirements.txt
└── main.py                   # 主入口文件


## 结构化输出

### OpenAI:

Supported models  支持的型号
Structured Outputs are available in our latest large language models, starting with GPT-4o:
结构化输出在我们最新的大型语言模型中可用，从GPT-4 o开始：

o3-mini-2025-1-31 and later
o3-mini-2025-1-31及更高版本
o1-2024-12-17 and later
o 1 -2024-12-17及以后
gpt-4o-mini-2024-07-18 and later
gpt-4 o-mini-2024-07-18及更高版本
gpt-4o-2024-08-06 and later
gpt-4 o-2024-08-06及更高版本
Older models like gpt-4-turbo and earlier may use JSON mode instead.
像gpt-4-turbo和更早的模型可能会使用[JSON模式]{https://platform.openai.com/docs/guides/structured-outputs#json-mode}。

Page: https://platform.openai.com/docs/guides/structured-outputs

### Qwen:

结构化输出功能支持以下模型：

qwen-max、qwen-plus、qwen-turbo以及qwen2.5系列模型。

Page: https://help.aliyun.com/zh/model-studio/user-guide/json-mode?spm=a2c4g.11186623.help-menu-search-2400256.d_0


