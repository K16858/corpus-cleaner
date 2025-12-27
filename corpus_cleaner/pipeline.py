"""3段階処理パイプライン"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from .cleaner import CorpusCleaner

try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False
    kenlm = None

try:
    from .perplexity import PerplexityCalculator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    PerplexityCalculator = None


class ProcessingPipeline:
    """3段階処理パイプライン"""
    
    def __init__(
        self,
        cleaner: CorpusCleaner,
        text_field: str = 'content',
        kenlm_model_path: Optional[str] = None,
        max_kenlm_perplexity: float = 100.0,
        use_llm: Optional[bool] = None,
        llm_model_name: str = "rinna/gemma-2-baku-2b",
        max_llm_perplexity: float = 100.0,
        auto_detect_models: bool = True
    ):
        """
        Args:
            cleaner: Phase 1用のCorpusCleanerインスタンス
            text_field: テキストが格納されているフィールド名
            kenlm_model_path: KenLMモデルファイルのパス（Phase 2、Noneの場合は自動検出を試みる）
            max_kenlm_perplexity: KenLMの最大perplexity値
            use_llm: LLMによる最終評価を使用するか（Noneの場合は自動判定）
            llm_model_name: LLMモデル名
            max_llm_perplexity: LLMの最大perplexity値
            auto_detect_models: モデルの自動検出を有効化するか
        """
        self.cleaner = cleaner
        self.text_field = text_field
        self.max_kenlm_perplexity = max_kenlm_perplexity
        self.max_llm_perplexity = max_llm_perplexity
        self.auto_detect_models = auto_detect_models
        
        self.kenlm_model = None
        if kenlm_model_path:
            if KENLM_AVAILABLE:
                try:
                    self.kenlm_model = kenlm.Model(kenlm_model_path)
                    print(f"KenLMモデルを読み込みました: {kenlm_model_path}")
                except Exception as e:
                    print(f"警告: KenLMモデルの読み込みに失敗しました: {e}")
                    print("KenLM処理はスキップされます。")
            else:
                print("警告: KenLMが利用できません。インストール: pip install kenlm")
        elif auto_detect_models and KENLM_AVAILABLE:
            import os
            common_paths = [
                'model.bin',
                'kenlm_model.bin',
                'lm.bin',
                os.path.expanduser('~/kenlm_model.bin'),
            ]
            for path in common_paths:
                if os.path.exists(path):
                    try:
                        self.kenlm_model = kenlm.Model(path)
                        print(f"KenLMモデルを自動検出して読み込みました: {path}")
                        break
                    except Exception:
                        continue
        
        self.llm_calculator = None
        if use_llm is True or (use_llm is None and auto_detect_models):
            if LLM_AVAILABLE:
                try:
                    self.llm_calculator = PerplexityCalculator(
                        model_name=llm_model_name
                    )
                    print(f"LLMモデルを読み込みました: {llm_model_name}")
                    self.use_llm = True
                except Exception as e:
                    print(f"警告: LLMモデルの読み込みに失敗しました: {e}")
                    print("LLM処理はスキップされます。")
                    self.use_llm = False
            else:
                self.use_llm = False
        else:
            self.use_llm = False
    
    def process_file(
        self,
        input_path: str,
        output_path: str,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        3段階処理を実行
        
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
        
        phase1_output = str(Path(output_path).parent / f"{Path(output_path).stem}_phase1.jsonl")
        print("=" * 60)
        print("Phase 1: 基本クリーニング処理")
        print("=" * 60)
        phase1_stats = self._phase1_basic_cleaning(input_path, phase1_output, show_progress)
        
        phase2_output = str(Path(output_path).parent / f"{Path(output_path).stem}_phase2.jsonl")
        if self.kenlm_model:
            print("\n" + "=" * 60)
            print("Phase 2: KenLMによる高速perplexity評価")
            print("=" * 60)
            phase2_stats = self._phase2_kenlm_filtering(phase1_output, phase2_output, show_progress)
        else:
            print("\nKenLMモデルが利用できないため、Phase 2をスキップします。")
            phase2_output = phase1_output
            phase2_stats = {}
        
        if self.use_llm and self.llm_calculator:
            print("\n" + "=" * 60)
            print("Phase 3: LLMによる最終評価")
            print("=" * 60)
            final_stats = self._phase3_llm_filtering(phase2_output, output_path, show_progress)
        else:
            print("\nLLM処理が無効または利用できないため、Phase 3をスキップします。")
            import shutil
            shutil.copy(phase2_output, output_path)
            final_stats = {}
        
        stats = {
            'phase1': phase1_stats,
            'phase2': phase2_stats,
            'phase3': final_stats,
        }
        
        return stats
    
    def _phase1_basic_cleaning(
        self,
        input_path: str,
        output_path: str,
        show_progress: bool
    ) -> Dict[str, Any]:
        """Phase 1: 基本クリーニング処理"""
        total_lines = self._count_lines(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            pbar = tqdm(total=total_lines, desc="Phase 1: 基本クリーニング", disable=not show_progress)
            total_processed = 0
            total_kept = 0
            
            try:
                for line in infile:
                    total_processed += 1
                    
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        self.cleaner.stats['json_decode_error'] += 1
                        pbar.update(1)
                        continue
                    
                    cleaned_entry = self.cleaner.clean(entry, self.text_field)
                    
                    if cleaned_entry is not None:
                        json.dump(cleaned_entry, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        total_kept += 1
                    
                    pbar.update(1)
            
            finally:
                pbar.close()
        
        stats = self.cleaner.get_stats()
        stats['total_processed'] = total_processed
        stats['total_kept'] = total_kept
        stats['total_excluded'] = total_processed - total_kept
        
        return stats
    
    def _phase2_kenlm_filtering(
        self,
        input_path: str,
        output_path: str,
        show_progress: bool
    ) -> Dict[str, Any]:
        """Phase 2: KenLMによる高速perplexity評価"""
        import math
        
        total_lines = self._count_lines(input_path)
        total_processed = 0
        total_kept = 0
        total_filtered = 0
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            pbar = tqdm(total=total_lines, desc="Phase 2: KenLM評価", disable=not show_progress)
            
            try:
                for line in infile:
                    total_processed += 1
                    
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        pbar.update(1)
                        continue
                    
                    text = entry.get(self.text_field, '')
                    if not text:
                        pbar.update(1)
                        continue
                    
                    try:
                        score = self.kenlm_model.score(text, bos=True, eos=True)
                        words = len(text.split())
                        if words > 0:
                            perplexity = math.exp(-score / words)
                            
                            if perplexity <= self.max_kenlm_perplexity:
                                json.dump(entry, outfile, ensure_ascii=False)
                                outfile.write('\n')
                                total_kept += 1
                            else:
                                total_filtered += 1
                        else:
                            total_filtered += 1
                    except Exception:
                        total_filtered += 1
                    
                    pbar.update(1)
            
            finally:
                pbar.close()
        
        return {
            'total_processed': total_processed,
            'total_kept': total_kept,
            'total_excluded': total_filtered
        }
    
    def _phase3_llm_filtering(
        self,
        input_path: str,
        output_path: str,
        show_progress: bool
    ) -> Dict[str, Any]:
        """Phase 3: LLMによる最終評価"""
        total_lines = self._count_lines(input_path)
        total_processed = 0
        total_kept = 0
        total_filtered = 0
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            pbar = tqdm(total=total_lines, desc="Phase 3: LLM評価", disable=not show_progress)
            
            try:
                for line in infile:
                    total_processed += 1
                    
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        pbar.update(1)
                        continue
                    
                    text = entry.get(self.text_field, '')
                    if not text:
                        pbar.update(1)
                        continue
                    
                    if self.llm_calculator.is_high_quality(text, self.max_llm_perplexity):
                        json.dump(entry, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        total_kept += 1
                    else:
                        total_filtered += 1
                    
                    pbar.update(1)
            
            finally:
                pbar.close()
        
        return {
            'total_processed': total_processed,
            'total_kept': total_kept,
            'total_excluded': total_filtered
        }
    
    def _count_lines(self, file_path: str) -> int:
        """ファイルの行数をカウント"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return -1

