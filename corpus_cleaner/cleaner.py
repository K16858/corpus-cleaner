"""クリーニング処理のメインロジック"""

import re
import unicodedata
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
        self.max_emoji_ratio = self.config.get('max_emoji_ratio', 0.1)
        self.max_repeat_chars = self.config.get('max_repeat_chars', 3)
        
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
        
        # 文字の正規化（最初に実行）
        text = self._normalize_text(text)
        
        # 正規化後のテキストでチェック
        if not self._check_length(text):
            self.stats['length_filtered'] += 1
            return None
        
        if not self._check_duplicate(text):
            self.stats['duplicate_filtered'] += 1
            return None
        
        if not self._check_impurities(text):
            self.stats['impurity_filtered'] += 1
            return None
        
        # すべてのチェックを通過 - 正規化されたテキストをエントリに反映
        entry[text_field] = text
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
        
        # 絵文字チェック
        if self._has_too_many_emojis(text):
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
    
    def _has_too_many_emojis(self, text: str) -> bool:
        """絵文字が多すぎるかチェック"""
        if not text:
            return False
        
        # 絵文字の検出（Unicode絵文字の範囲）
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "]+",
            flags=re.UNICODE
        )
        
        emoji_count = len(emoji_pattern.findall(text))
        total_length = len(text)
        
        if total_length == 0:
            return False
        
        emoji_ratio = emoji_count / total_length
        return emoji_ratio > self.max_emoji_ratio
    
    def _normalize_text(self, text: str) -> str:
        """
        テキストの正規化
        
        Args:
            text: 正規化前のテキスト
            
        Returns:
            正規化されたテキスト
        """
        # 1. 全角英数字の半角変換
        text = self._convert_fullwidth_to_halfwidth(text)
        
        # 2. 改行の正規化
        text = self._normalize_newlines(text)
        
        # 3. 崩れた表記の正規化
        text = self._normalize_broken_notation(text)
        
        return text
    
    def _convert_fullwidth_to_halfwidth(self, text: str) -> str:
        """全角英数字を半角に変換"""
        # 全角英字（Ａ-Ｚ、ａ-ｚ）を半角に変換
        text = text.translate(str.maketrans(
            'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
            'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
            '０１２３４５６７８９',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            'abcdefghijklmnopqrstuvwxyz'
            '0123456789'
        ))
        
        # 全角スペースを半角スペースに変換
        text = text.replace('　', ' ')
        
        return text
    
    def _normalize_newlines(self, text: str) -> str:
        """改行の正規化"""
        # CRLFをLFに統一
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # 連続する改行を2つまでに制限（段落区切りとして保持）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 文の途中の改行を除去（簡易版）
        # 句点・読点・感嘆符・疑問符の後以外の改行をスペースに変換
        # ただし、見出し記号（#、##など）の前後は保持
        lines = text.split('\n')
        normalized_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                # 空行は保持（段落区切りとして）
                if normalized_lines and normalized_lines[-1]:
                    normalized_lines.append('')
                continue
            
            # 見出し記号で始まる行は保持
            if re.match(r'^#{1,6}\s+', line):
                normalized_lines.append(line)
                continue
            
            # 前の行が句点・読点・感嘆符・疑問符で終わっている場合は改行を保持
            if normalized_lines and normalized_lines[-1]:
                prev_line = normalized_lines[-1]
                if re.search(r'[。、！？]$', prev_line):
                    normalized_lines.append(line)
                else:
                    # 前の行に続ける（スペースで結合）
                    normalized_lines[-1] = prev_line + ' ' + line
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _normalize_broken_notation(self, text: str) -> str:
        """崩れた表記の正規化"""
        # 過剰な繰り返し文字の正規化（3回以上を制限）
        # 例: "wwww" -> "www", "！！！" -> "！！！"（3回まで）
        text = re.sub(r'(.)\1{' + str(self.max_repeat_chars) + r',}', 
                     r'\1' * self.max_repeat_chars, text)
        
        # アスキーアートの簡易検出と除去
        # 装飾的な文字パターン（連続する特殊文字や記号）
        ascii_art_patterns = [
            r'[─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏]{3,}',  # 罫線文字
            r'[═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬]{3,}',  # 罫線文字2
        ]
        
        for pattern in ascii_art_patterns:
            text = re.sub(pattern, '', text)
        
        # 絵文字が多すぎる場合は除外（チェックは後で行う）
        # ここでは正規化のみ
        
        return text
    
    def get_stats(self) -> Dict[str, int]:
        """統計情報を取得"""
        return dict(self.stats)

