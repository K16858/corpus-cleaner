"""クリーニング処理のメインロジック"""

import re
from typing import Dict, Any, Optional, Set
from html.parser import HTMLParser
from bs4 import BeautifulSoup
from collections import defaultdict


class CorpusCleaner:
    """コーパスクリーナークラス"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: クリーニング設定
        """
        self.config = config or {}
        
        # 設定のデフォルト値
        self.min_length = self.config.get('min_length', 10)
        self.max_length = self.config.get('max_length', 10000)
        self.max_special_char_ratio = self.config.get('max_special_char_ratio', 0.3)
        self.max_code_ratio = self.config.get('max_code_ratio', 0.2)
        self.max_html_ratio = self.config.get('max_html_ratio', 0.2)
        
        # 重複検出用のセット
        self.seen_texts: Set[str] = set()
        
        # 統計情報
        self.stats = defaultdict(int)
    
    def clean(self, entry: Dict[str, Any], text_field: str = 'text') -> Optional[Dict[str, Any]]:
        """
        エントリをクリーニング
        
        Args:
            entry: JSONLエントリ
            text_field: テキストが格納されているフィールド名
            
        Returns:
            クリーニング済みエントリ、または除外された場合はNone
        """
        if text_field not in entry:
            self.stats['missing_text_field'] += 1
            return None
        
        text = entry[text_field]
        if not isinstance(text, str):
            self.stats['invalid_text_type'] += 1
            return None
        
        # 各チェックを実行
        if not self._check_length(text):
            self.stats['length_filtered'] += 1
            return None
        
        if not self._check_duplicate(text):
            self.stats['duplicate_filtered'] += 1
            return None
        
        if not self._check_impurities(text):
            self.stats['impurity_filtered'] += 1
            return None
        
        # すべてのチェックを通過
        self.stats['kept'] += 1
        return entry
    
    def _check_length(self, text: str) -> bool:
        """長さチェック"""
        length = len(text)
        if length < self.min_length:
            return False
        if length > self.max_length:
            return False
        return True
    
    def _check_duplicate(self, text: str) -> bool:
        """重複チェック"""
        # 正規化（空白を統一）
        normalized = re.sub(r'\s+', ' ', text.strip())
        
        if normalized in self.seen_texts:
            return False
        
        self.seen_texts.add(normalized)
        return True
    
    def _check_impurities(self, text: str) -> bool:
        """不純物チェック"""
        # HTMLチェック
        if self._has_too_much_html(text):
            return False
        
        # コードチェック
        if self._has_too_much_code(text):
            return False
        
        # ログチェック
        if self._looks_like_log(text):
            return False
        
        # 特殊記号チェック
        if self._has_too_many_special_chars(text):
            return False
        
        return True
    
    def _has_too_much_html(self, text: str) -> bool:
        """HTMLタグが多すぎるかチェック"""
        soup = BeautifulSoup(text, 'html.parser')
        html_tags = soup.find_all()
        
        if not html_tags:
            return False
        
        # HTMLタグの文字数を計算
        html_text_length = sum(len(tag.get_text()) for tag in html_tags)
        total_length = len(text)
        
        if total_length == 0:
            return False
        
        html_ratio = html_text_length / total_length
        return html_ratio > self.max_html_ratio
    
    def _has_too_much_code(self, text: str) -> bool:
        """コードが多すぎるかチェック"""
        # コードらしいパターンを検出
        code_patterns = [
            r'def\s+\w+\s*\(',  # 関数定義
            r'class\s+\w+',  # クラス定義
            r'import\s+\w+',  # import文
            r'#include\s*<',  # C/C++ include
            r'function\s+\w+\s*\(',  # JavaScript関数
            r'const\s+\w+\s*=',  # const宣言
            r'let\s+\w+\s*=',  # let宣言
            r'var\s+\w+\s*=',  # var宣言
            r'<\?php',  # PHP
            r'<\?=',  # PHP短縮タグ
            r'```',  # コードブロック
            r'```\w+',  # 言語指定付きコードブロック
        ]
        
        code_char_count = 0
        total_length = len(text)
        
        if total_length == 0:
            return False
        
        for pattern in code_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # マッチした行の長さを加算（簡易的な実装）
                line_start = text.rfind('\n', 0, match.start())
                line_end = text.find('\n', match.end())
                if line_end == -1:
                    line_end = len(text)
                code_char_count += (line_end - line_start)
        
        code_ratio = code_char_count / total_length
        return code_ratio > self.max_code_ratio
    
    def _looks_like_log(self, text: str) -> bool:
        """ログファイルらしいかチェック"""
        # ログパターン
        log_patterns = [
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # タイムスタンプ
            r'\[(DEBUG|INFO|WARN|ERROR|FATAL)\]',  # ログレベル
            r'ERROR:',  # エラーメッセージ
            r'Exception:',  # 例外
            r'Traceback\s+\(most recent call last\)',  # Python traceback
        ]
        
        # パターンが多くマッチする場合はログと判断
        match_count = sum(1 for pattern in log_patterns if re.search(pattern, text))
        
        # 3つ以上のパターンがマッチしたらログと判断
        return match_count >= 3
    
    def _has_too_many_special_chars(self, text: str) -> bool:
        """特殊記号が多すぎるかチェック"""
        if not text:
            return False
        
        # 特殊記号の定義（日本語テキストに通常含まれない記号）
        special_chars = set('!@#$%^&*()_+-=[]{}|;:,.<>?/~`')
        
        special_count = sum(1 for char in text if char in special_chars)
        total_length = len(text)
        
        special_ratio = special_count / total_length
        return special_ratio > self.max_special_char_ratio
    
    def get_stats(self) -> Dict[str, int]:
        """統計情報を取得"""
        return dict(self.stats)

