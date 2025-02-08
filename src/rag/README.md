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
