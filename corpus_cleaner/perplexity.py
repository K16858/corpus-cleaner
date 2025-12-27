"""Perplexity計算による品質評価"""

import re
import math
from typing import Optional, List

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


class PerplexityCalculator:
    """Perplexity計算クラス"""
    
    def __init__(
        self,
        model_name: str = "rinna/gemma-2-baku-2b",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 512
    ):
        """
        Args:
            model_name: 使用するモデル名（デフォルト: rinna/gemma-2-baku-2b）
            device: 使用するデバイス（Noneの場合は自動選択）
            batch_size: バッチサイズ
            max_length: 最大トークン長
        """
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # デバイスの設定
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """モデルを読み込む（遅延読み込み）"""
        if self._model_loaded:
            return
        
        import time
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.model.to(self.device)
                self.model.eval()
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self._model_loaded = True
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"モデルの読み込みに失敗しました（試行 {attempt + 1}/{max_retries}）。{retry_delay}秒後に再試行します...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"モデルの読み込みに失敗しました（{max_retries}回試行）: {e}")
    
    def calculate_perplexity(self, text: str) -> Optional[float]:
        """
        テキストのperplexityを計算
        
        Args:
            text: 評価するテキスト
            
        Returns:
            Perplexity値（計算に失敗した場合はNone）
        """
        if not text or not text.strip():
            return None
        
        # モデルが読み込まれていない場合は読み込む
        if not self._model_loaded:
            self._load_model()
        
        try:
            # テキストをトークン化
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)
            
            # Perplexityを計算
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexity = math.exp(loss)
            
            return perplexity
        
        except Exception as e:
            # 計算に失敗した場合はNoneを返す
            return None
    
    def calculate_perplexity_batch(self, texts: List[str]) -> List[Optional[float]]:
        """
        複数のテキストのperplexityをバッチで計算
        
        Args:
            texts: 評価するテキストのリスト
            
        Returns:
            Perplexity値のリスト（計算に失敗した場合はNone）
        """
        if not texts:
            return []
        
        # モデルが読み込まれていない場合は読み込む
        if not self._model_loaded:
            self._load_model()
        
        results = []
        
        # バッチ処理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = []
            
            for text in batch_texts:
                perplexity = self.calculate_perplexity(text)
                batch_results.append(perplexity)
            
            results.extend(batch_results)
        
        return results
    
    def is_high_quality(self, text: str, max_perplexity: float = 100.0) -> bool:
        """
        テキストが高品質かどうかを判定
        
        Args:
            text: 評価するテキスト
            max_perplexity: 許容される最大perplexity値
            
        Returns:
            高品質な場合はTrue、そうでない場合はFalse
        """
        perplexity = self.calculate_perplexity(text)
        
        if perplexity is None:
            # 計算に失敗した場合は除外
            return False
        
        return perplexity <= max_perplexity

