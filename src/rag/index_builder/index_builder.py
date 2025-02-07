from typing import List, Dict, Optional
import os
import logging
import json
import uuid
import re
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from .vectorstore_manager import VectorstoreManager

class CustomDocument:
    def __init__(self, metadata: Dict[str, str], page_content: str):
        self.id = str(uuid.uuid4())  # Generate unique id
        self.metadata = metadata
        self.page_content = page_content

    def __repr__(self):
        return f"Document(metadata={self.metadata}, page_content='{self.page_content}')"

    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format"""
        return Document(page_content=self.page_content, metadata=self.metadata)

class IndexBuilder:
    def __init__(self, config: Dict):
        """
        Initialize index builder class
        Args:
            config: Configuration dictionary containing vector store settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["vectorstore"]["embedding_model"]
        )
        
        self.metadata_path = None
        if config["vectorstore"]["metadata_path"]:
            self.metadata_path = config["vectorstore"]["metadata_path"]
        
        if config["vectorstore"]["persist_directory"]:
            self.persist_directory = config["vectorstore"]["persist_directory"]
        else:
            raise ValueError("arg persist_directory is required")
        
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        if config["vectorstore"]["embedding_model"]:
            self.embedding_model = config["vectorstore"]["embedding_model"]
            
        self.vectorstore_manager = VectorstoreManager(config)

    def extract_from_json_files(self, files_path: List[str]) -> List[CustomDocument]:
        """
        Extract documents from JSON files in the specified directory
        如果是文件夹，则读取文件夹中的所有文件；如果是指定的几个文件，那么读取指定的这几个文件。
        Args:
            files_path: Directory containing JSON files
        Returns:
            List of Document objects
        """
        if len(files_path) == 1:
            # 文件夹路径，读取文件夹中的所有文件
            one_file_paths =[]
            for file_name in os.listdir(files_path[0]):
                one_file_paths.append(os.path.join(files_path[0], file_name))
            
        if len(files_path) > 1:
            # 读取指定的几个文件
            one_file_paths =[]
            for file_name in files_path:
                one_file_paths.append(file_name)
        
        all_documents: List[CustomDocument] = []
        for one_filepath in one_file_paths:
            if not one_filepath.endswith('.json'):
                continue
            self.logger.info(f"Reading file: {one_filepath.split('/')[-1]}")
            try:
                with open(one_filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    data = data[1:]  # Skip first entry (usually metadata or catalogue)
                    for entry in data:
                        if "output" in entry:
                            page_content = entry["output"].get("scene", "")    # 建立索引的内容
                            file_name_specific = entry.get("metadata", {}).get("file_name", "unknown source")
                            element_id = entry.get("metadata", {}).get("element_id", 0)
                            languages = entry.get("metadata", {}).get("languages", [])
                            if not isinstance(languages, list) or len(languages) != 1 or not all(isinstance(lang, str) for lang in languages):
                                self.logger.warning(f"Invalid languages format in entry: {languages}. Faction vectorstore.add_documents() expected metadata value to be a str, int, float or bool.")
                            languages_str = languages[0]
                            text = entry.get("text", {})
                            
                            # format the output
                            output = entry.get("output", {})
                            output_intro = output.get("intro", {})
                            output_personalities_trails = output.get("personalities_trails", {})
                            output_self_awareness = output.get("self_awareness", {})
                            output_scene = output.get("scene", {})
                            
                            # Chroma not support dict, so we need find info with original files.
                            if "output-short" in entry:
                                output_short = True
                            else:
                                output_short = False
                                
                            document = CustomDocument(
                                metadata={
                                    "original_text": text,
                                    "file_name": file_name_specific,
                                    "element_id": element_id,
                                    "languages": languages_str,
                                    "output-intro": output_intro,
                                    "output-personalities_trails": output_personalities_trails,
                                    "output-self_awareness": output_self_awareness,
                                    "output-scene": output_scene,
                                    "exist-output-short": output_short
                                },
                                page_content=page_content
                            )
                            all_documents.append(document)
                            
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON file '{one_filepath}': {e}")
                
        return all_documents

    def build_index(self, 
                   initialize_meta_data: bool = False, 
                   files: List[str] = [],
                   persist_directory: Optional[str] = None,
                   batch_size: int = 512) -> Chroma:
        """
        Build index with batch processing
        Args:
            initialize_meta_data: 如果是第一次运行，需要将元数据初始化为向量存储。
            files: List[str]当前角色所涉及到的文件
            documents: List of documents
            persist_directory: Directory to persist the vector store
            batch_size: Size of batches for processing
            metadata_path: Description of the first initialization parameter
            embedding_model: Description of the second initialization parameter
        Returns:
            Built vector store
        """
        
        # 设定文件的读取路径
        files_path = []
        if initialize_meta_data:
            files_path.append(self.metadata_path)    
            persist_directory = os.path.join(self.persist_directory, self.metadata_path.split('/')[-1])
        elif not initialize_meta_data and len(files) > 0:
            files_path = files
            # Generate persist directory name based on file names
            combined_name = self.vectorstore_manager.generate_persist_directory(files)
            persist_directory = os.path.join(self.persist_directory, combined_name)
        else:
            raise ValueError("arg files is required")
            

        
        all_splits = self.extract_from_json_files(files_path=files_path)
        all_splits_length = len(all_splits)
        logging.info(f"Beginning generate vector store, the splits size is {all_splits_length}.")
        
        # Initialize vector store without documents
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Process documents in batches
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            vectorstore.add_documents(batch)  # 将当前批次添加到 Chroma 中
            print(f"已处理分块数量 {i} -> {i + len(batch)}.")
        
        num_vectors = vectorstore._collection.count()
        self.logger.info(f"Number of vectors stored: {num_vectors}")
        self.logger.info(f"Successfully built index, stored in {persist_directory}")
        
        return vectorstore


    def load_index(self, persist_directory: str) -> Optional[Chroma]:
        """
        Load existing index
        Args:
            persist_directory: Directory where the vector store is persisted
        Returns:
            Loaded vector store, or None if it doesn't exist
        """
        if not os.path.exists(persist_directory):
            self.logger.warning(f"Index directory {persist_directory} does not exist")
            self.logger.info(f"新建 metadata 索引 {persist_directory} ...")
            vectorstore = self.build_index(initialize_meta_data=True)
            
            return vectorstore

        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            self.logger.info(f"Succeed to load index.")
            
            return vectorstore
        
        except Exception as e:
            self.logger.error(f"Failed to load index: {str(e)}")
            
            raise

    def update_index(self, 
                    persist_directory: str, 
                    new_documents: List[CustomDocument],
                    batch_size: int = 512) -> Optional[Chroma]:
        """
        Update existing index
        Args:
            persist_directory: Directory where the vector store is persisted
            new_documents: List of new documents
            batch_size: Size of batches for processing
        Returns:
            Updated vector store
        """
        vectorstore = self.load_index(persist_directory)
        if vectorstore is None:
            return self.build_index(new_documents, persist_directory, batch_size)

        try:
            # Process new documents in batches
            total_docs = len(new_documents)
            for i in range(0, total_docs, batch_size):
                batch = new_documents[i:i + batch_size]
                # Convert CustomDocuments to LangChain Documents
                langchain_docs = [doc.to_langchain_document() for doc in batch]
                vectorstore.add_documents(langchain_docs)
                self.logger.info(f"Processed new documents {i} to {i + len(batch)}")

            vectorstore.persist()
            self.logger.info(f"Successfully updated index in {persist_directory}")
            return vectorstore
        except Exception as e:
            self.logger.error(f"Failed to update index: {str(e)}")
            raise 