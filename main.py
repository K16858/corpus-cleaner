#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from corpus_cleaner.cleaner import CorpusCleaner
from corpus_cleaner.pipeline import ProcessingPipeline

def main():
    parser = argparse.ArgumentParser(
        prog='corpus_cleaner'
    )
    parser.add_argument(
        'input',
        type=str,
        help='入力JSONLファイルのパス'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='出力JSONLファイルのパス（デフォルト: input_cleaned.jsonl）'
    )
    parser.add_argument(
        '--text-field',
        type=str,
        default='content',
        help='テキストが格納されているフィールド名（デフォルト: content）'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='最小文字数（デフォルト: 10）'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=10000,
        help='最大文字数（デフォルト: 10000）'
    )
    parser.add_argument(
        '--max-special-char-ratio',
        type=float,
        default=0.3,
        help='特殊記号の最大比率（デフォルト: 0.3）'
    )
    parser.add_argument(
        '--max-code-ratio',
        type=float,
        default=0.2,
        help='コードの最大比率（デフォルト: 0.2）'
    )
    parser.add_argument(
        '--max-html-ratio',
        type=float,
        default=0.2,
        help='HTMLの最大比率（デフォルト: 0.2）'
    )
    parser.add_argument(
        '--max-emoji-ratio',
        type=float,
        default=0.1,
        help='絵文字の最大比率（デフォルト: 0.1）'
    )
    parser.add_argument(
        '--max-repeat-chars',
        type=int,
        default=3,
        help='繰り返し文字の最大回数（デフォルト: 3）'
    )
    parser.add_argument(
        '--max-sentence-length',
        type=int,
        default=500,
        help='1文の最大文字数（デフォルト: 500）'
    )
    parser.add_argument(
        '--require-sentence-end',
        action='store_true',
        default=True,
        help='文末記号（。！？）で終わることを要求（デフォルト: True）'
    )
    parser.add_argument(
        '--no-require-sentence-end',
        dest='require_sentence_end',
        action='store_false',
        help='文末記号の要求を無効化'
    )
    parser.add_argument(
        '--min-sentence-end-ratio',
        type=float,
        default=0.7,
        help='文末記号で終わる文の最小比率（デフォルト: 0.7）'
    )
    parser.add_argument(
        '--min-hiragana-ratio',
        type=float,
        default=0.3,
        help='ひらがなの最小比率（デフォルト: 0.3）'
    )
    parser.add_argument(
        '--max-hiragana-ratio',
        type=float,
        default=0.8,
        help='ひらがなの最大比率（デフォルト: 0.8）'
    )
    parser.add_argument(
        '--min-kanji-ratio',
        type=float,
        default=0.1,
        help='漢字の最小比率（デフォルト: 0.1）'
    )
    parser.add_argument(
        '--max-kanji-ratio',
        type=float,
        default=0.5,
        help='漢字の最大比率（デフォルト: 0.5）'
    )
    # Phase 2: KenLM設定
    parser.add_argument(
        '--kenlm-model',
        type=str,
        default=None,
        help='KenLMモデルファイルのパス（Phase 2で使用、.bin形式、未指定の場合は自動検出を試みる）'
    )
    parser.add_argument(
        '--sentencepiece-model',
        type=str,
        default=None,
        help='SentencePieceモデルファイルのパス（.model形式、cc_netのja.sp.modelなど）'
    )
    parser.add_argument(
        '--no-kenlm',
        action='store_true',
        default=False,
        help='KenLM処理を無効化'
    )
    parser.add_argument(
        '--max-kenlm-perplexity',
        type=float,
        default=100.0,
        help='KenLMの最大perplexity値（デフォルト: 100.0）'
    )
    # Phase 3: LLM設定
    parser.add_argument(
        '--use-llm',
        action='store_true',
        default=None,
        help='LLMによる最終評価を有効化（Phase 3、未指定の場合は自動検出）'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        default=False,
        help='LLM処理を無効化'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='rinna/gemma-2-baku-2b',
        help='LLMモデル名（デフォルト: rinna/gemma-2-baku-2b）'
    )
    parser.add_argument(
        '--max-llm-perplexity',
        type=float,
        default=10.0,
        help='LLMの最大perplexity値（デフォルト: 10.0、値が小さいほど厳格）'
    )
    parser.add_argument(
        '--no-auto-detect',
        action='store_true',
        default=False,
        help='モデルの自動検出を無効化'
    )
    parser.add_argument(
        '--stats-output',
        type=str,
        default=None,
        help='統計情報を出力するJSONファイルのパス（デフォルト: statistics.json）'
    )
    
    args = parser.parse_args()
    
    # 出力ファイルパスの決定
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_cleaned.jsonl")
    
    # 統計ファイルパスの決定
    if args.stats_output is None:
        args.stats_output = "statistics.json"
    
    # 設定の作成
    config = {
        'min_length': args.min_length,
        'max_length': args.max_length,
        'max_special_char_ratio': args.max_special_char_ratio,
        'max_code_ratio': args.max_code_ratio,
        'max_html_ratio': args.max_html_ratio,
        'max_emoji_ratio': args.max_emoji_ratio,
        'max_repeat_chars': args.max_repeat_chars,
        'max_sentence_length': args.max_sentence_length,
        'require_sentence_end': args.require_sentence_end,
        'min_sentence_end_ratio': args.min_sentence_end_ratio,
        'min_hiragana_ratio': args.min_hiragana_ratio,
        'max_hiragana_ratio': args.max_hiragana_ratio,
        'min_kanji_ratio': args.min_kanji_ratio,
        'max_kanji_ratio': args.max_kanji_ratio,
    }
    
    # クリーナーとパイプラインの作成
    cleaner = CorpusCleaner(config)
    
    # KenLMモデルパスの決定（--no-kenlmが指定されている場合はNone）
    kenlm_model_path = None if args.no_kenlm else args.kenlm_model
    
    # LLM使用の決定（--no-llmが指定されている場合はFalse、--use-llmが指定されている場合はTrue、それ以外はNoneで自動判定）
    use_llm = False if args.no_llm else (True if args.use_llm else None)
    
    pipeline = ProcessingPipeline(
        cleaner=cleaner,
        text_field=args.text_field,
        kenlm_model_path=kenlm_model_path,
        sentencepiece_model_path=args.sentencepiece_model,
        max_kenlm_perplexity=args.max_kenlm_perplexity,
        use_llm=use_llm,
        llm_model_name=args.llm_model,
        max_llm_perplexity=args.max_llm_perplexity,
        auto_detect_models=not args.no_auto_detect
    )
    
    print(f"入力ファイル: {args.input}")
    print(f"出力ファイル: {args.output}")
    print(f"テキストフィールド: {args.text_field}")
    print()
    
    try:
        # 3段階処理を実行
        stats = pipeline.process_file(args.input, args.output)
        
        # 統計情報の出力
        with open(args.stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 結果の表示
        print("\n" + "=" * 60)
        print("最終処理結果")
        print("=" * 60)
        
        # Phase 1の結果
        if 'phase1' in stats:
            phase1 = stats['phase1']
            print(f"\nPhase 1 (基本クリーニング):")
            print(f"  総処理数: {phase1.get('total_processed', 0):,}")
            print(f"  保持数: {phase1.get('total_kept', 0):,}")
            print(f"  除外数: {phase1.get('total_excluded', 0):,}")
            if phase1.get('total_processed', 0) > 0:
                keep_ratio = phase1.get('total_kept', 0) / phase1.get('total_processed', 1) * 100
                print(f"  保持率: {keep_ratio:.2f}%")
        
        # Phase 2の結果
        if 'phase2' in stats and stats['phase2']:
            phase2 = stats['phase2']
            print(f"\nPhase 2 (KenLM評価):")
            print(f"  総処理数: {phase2.get('total_processed', 0):,}")
            print(f"  保持数: {phase2.get('total_kept', 0):,}")
            print(f"  除外数: {phase2.get('total_excluded', 0):,}")
            if phase2.get('total_processed', 0) > 0:
                keep_ratio = phase2.get('total_kept', 0) / phase2.get('total_processed', 1) * 100
                print(f"  保持率: {keep_ratio:.2f}%")
        
        # Phase 3の結果
        if 'phase3' in stats and stats['phase3']:
            phase3 = stats['phase3']
            print(f"\nPhase 3 (LLM評価):")
            print(f"  総処理数: {phase3.get('total_processed', 0):,}")
            print(f"  保持数: {phase3.get('total_kept', 0):,}")
            print(f"  除外数: {phase3.get('total_excluded', 0):,}")
            if phase3.get('total_processed', 0) > 0:
                keep_ratio = phase3.get('total_kept', 0) / phase3.get('total_processed', 1) * 100
                print(f"  保持率: {keep_ratio:.2f}%")
        
        print()
        print(f"統計情報を保存しました: {args.stats_output}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
