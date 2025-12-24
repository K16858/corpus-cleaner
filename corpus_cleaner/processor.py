"""JSONLファイルのストリーミング処理"""

import json
import sys
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm

from .cleaner import TextCleaner


class JSONLProcessor:
    """JSONLファイルのストリーミング処理クラス"""
    
    def __init__(self, cleaner: TextCleaner, text_field: str = 'text'):
        """
        Args:
            cleaner: TextCleanerインスタンス
            text_field: テキストが格納されているフィールド名
        """
        self.cleaner = cleaner
        self.text_field = text_field
        self.total_processed = 0
        self.total_kept = 0
    
    def process_file(
        self,
        input_path: str,
        output_path: str,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        JSONLファイルを処理
        
        Args:
            input_path: 入力JSONLファイルのパス
            output_path: 出力JSONLファイルのパス
            show_progress: プログレスバーを表示するか
            
        Returns:
            処理統計情報
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")
        
        # 総行数を取得（プログレスバー用）
        total_lines = self._count_lines(input_path)
        
        # ストリーミング処理
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            pbar = tqdm(total=total_lines, desc="処理中", disable=not show_progress)
            
            try:
                for line in infile:
                    self.total_processed += 1
                    
                    # JSONパース
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        self.cleaner.stats['json_decode_error'] += 1
                        pbar.update(1)
                        continue
                    
                    # クリーニング
                    cleaned_entry = self.cleaner.clean(entry, self.text_field)
                    
                    if cleaned_entry is not None:
                        # 出力
                        json.dump(cleaned_entry, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        self.total_kept += 1
                    
                    pbar.update(1)
            
            finally:
                pbar.close()
        
        # 統計情報をまとめる
        stats = self.cleaner.get_stats()
        stats['total_processed'] = self.total_processed
        stats['total_kept'] = self.total_kept
        stats['total_excluded'] = self.total_processed - self.total_kept
        
        return stats
    
    def _count_lines(self, file_path: str) -> int:
        """ファイルの行数をカウント（簡易版）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return -1
