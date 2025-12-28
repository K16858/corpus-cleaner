# text_cleaner
日本語コーパスのJSONLファイルをクリーニングするツールです。

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py <入力JSONLファイル> -o <出力JSONLファイル>
```

例：

```bash
python main.py data/input.jsonl -o data/output_cleaned.jsonl
```

詳細なオプションは `python main.py --help` で確認できます。

## KenLMのモデルダウンロード
クリーニングのフェーズ2にKenLMモデルが必要です。
以下のコマンドでダウンロードしてください。

```bash
# modelディレクトリにモデルを配置
mkdir -p model
# KenLMモデルをダウンロード
wget -c  -P data/lm_sp http://dl.fbaipublicfiles.com/cc_net/lm/ja.arpa.bin
wget -c  -P data/lm_sp http://dl.fbaipublicfiles.com/cc_net/lm/ja.sp.model
```

モデルは別のものでも構いません。
