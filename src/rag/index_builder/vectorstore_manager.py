from typing import Dict, List
import os
import json
import re
import logging

class VectorstoreManager:
    def __init__(self, config: Dict):
        """
        Initialize vectorstore manager, 只有在类实例化的时候才会读一次数据。因此如果需要使用 mapping 数据，需要使用 _load_mappings 方法拿到最新的数据。
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.mapping_file = 'data/vectorstore/vectorstore_mappings.json'
        self.mappings = self._load_mappings()
        self._validate_mappings()
        
    # FIX 需要验证
    def _validate_mappings(self):
        """
        Validate that all files in mappings exist in the data directory.
        If any files are missing, log a warning message.
        
        return True or False
        """
        if self.mappings == {}:
            return True
            
        missing_files = []
        for persist_dir, files in self.mappings.items():
            for file in files:
                file_path = os.path.join('data', file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
        
        if missing_files:
            warning_msg = (
                "警告：vectorstore_mappings 中的以下文件不存在：\n"
                f"{json.dumps(missing_files, indent=2, ensure_ascii=False)}\n"
                "请检查 data/ 下的所有文件是否正确。如不正确，请删除 data/vectorstore/ 下的文件，重新构建向量存储。"
            )
            self.logger.warning(warning_msg)
            
    def _load_mappings(self) -> Dict:
        """Load existing mappings or create new ones"""
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    self.logger.warning("vectorstore_mappings.json 文件为空，返回 {} 替代")
                    return {}
                return json.load(f)
        
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.mapping_file), exist_ok=True)
            # Create an empty JSON file
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                pass  # 不写入任何内容，只创建文件
            return {}
    
    def _save_mappings(self):
        """Save mappings to file by appending"""
        # Load existing mappings
        existing_mappings = {}
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                existing_mappings = json.load(f)
        
        # Update with new mappings
        existing_mappings.update(self.mappings)
        
        # Write back all mappings
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(existing_mappings, f, indent=2, ensure_ascii=False)
    
    def generate_persist_directory(self, files: List[str]) -> str:
        """
        Generate a unique persist directory name from file names
        Args:
            files: List of file names
        Returns:
            Persist directory name
        """
        # Extract book names without extensions and special characters
        book_names = []
        for file in files:
            # Remove file extension and special characters
            name = os.path.splitext(file)[0]
            # Extract the main book name (before "traits" or other identifiers)
            match = re.match(r'traits_(.*?)(?:_CH.*|$)', name)
            if match:
                book_name = match.group(1)
                book_names.append(book_name)
        
        # Join book names and create a unique identifier
        combined_name = '_'.join(sorted(book_names))
        
        # Update mappings
        if combined_name not in self.mappings:
            self.mappings[combined_name] = files
            self._save_mappings()
            
        return combined_name
    
    def get_files_for_persist_directory(self, persist_directory: str) -> List[str]:
        """
        Get the list of files associated with a persist directory
        Args:
            persist_directory: The persist directory name
        Returns:
            List of associated files
        """
        return self.mappings.get(persist_directory, []) 