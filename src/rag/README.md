# rag

project/
├── config/
│   └── config.yaml           # 配置文件
├── llm_api/
│   ├── **init**.py
│   ├── llm_api.py           # LLMAPI 类
│   ├── query_rewriting.py   # QueryRewriting 类
│   └── [utils.py](http://utils.py/)             # 通用工具函数
├── role_determination/
│   ├── **init**.py
│   └── role_determination.py # RoleDetermination 类
├── index_builder/
│   ├── **init**.py
│   └── index_builder.py      # IndexBuilder 类
├── rag_service/
│   ├── **init**.py
│   └── rag_service.py        # RAGService 类
├── data/
│   └── vectorstore/          # 存放向量数据库文件
├── tests/
│   ├── **init**.py
│   ├── test_llm_api.py
│   ├── test_role_determination.py
│   ├── test_index_builder.py
│   └── test_rag_service.py
├── requirements.txt
└── [main.py](http://main.py/)                   # 主入口文件