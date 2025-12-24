#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from corpus_cleaner.cleaner import CorpusCleaner
from corpus_cleaner.processor import JSONLProcessor

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
    
    # クリーナーとプロセッサーの作成
    cleaner = CorpusCleaner(config)
    processor = JSONLProcessor(cleaner, args.text_field)
    
    print(f"入力ファイル: {args.input}")
    print(f"出力ファイル: {args.output}")
    print(f"テキストフィールド: {args.text_field}")
    print()
    
    try:
        # 処理実行
        stats = processor.process_file(args.input, args.output)
        
        # 統計情報の出力
        with open(args.stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 結果の表示
        print()
        print("=" * 60)
        print("処理結果")
        print("=" * 60)
        print(f"総処理数: {stats['total_processed']:,}")
        print(f"保持数: {stats['total_kept']:,}")
        print(f"除外数: {stats['total_excluded']:,}")
        if stats['total_processed'] > 0:
            keep_ratio = stats['total_kept'] / stats['total_processed'] * 100
            print(f"保持率: {keep_ratio:.2f}%")
        print()
        print("除外理由:")
        for key, value in stats.items():
            if key not in ['total_processed', 'total_kept', 'total_excluded', 'kept'] and value > 0:
                print(f"  {key}: {value:,}")
        print()
        print(f"統計情報を保存しました: {args.stats_output}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
