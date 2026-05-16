# tilavet-aligner 0.1.0 — Release & Publication Plan

İlk public release. Phonemizer 0.2.0'i tamamlayan ikinci ayak.

## Build Durumu (Mayis 2026)

- 35 unit test geciyor (1 skip), Python 3.9+ destegi.
- `python3 -m build` temiz: 15 KB whl + 17 KB sdist.
- `twine check`: PASSED.
- Venv install smoke: `tilavet-align --help` + toy alignment cikti dogru.
- Bagimliliklar: zero runtime deps (numpy opsiyonel `posterior` extra'sinda).

## Yeni Repo Kurulum (GitHub'da yoksa)

Phonemizer ile ayni hesap altinda `tilavet-aligner` adinda yeni public repo
ac. Ardindan local repo'yu init et ve push'la:

```bash
cd "/Users/baris/Documents/GitHub/Tilavet/tilavet-aligner"
git init
git branch -m main
git add -A
git commit -m "initial commit: tilavet-aligner 0.1.0"
git remote add origin https://github.com/tialkan/tilavet-aligner.git
git push -u origin main
```

## GitHub Tag + Release

```bash
git tag -a v0.1.0 -m "tilavet-aligner 0.1.0 — first public release"
git push origin v0.1.0
```

GitHub release notlari icin `CHANGELOG.md`'deki `[0.1.0]` bolumunu kopyala.
Release URL:

```
https://github.com/tialkan/tilavet-aligner/releases/new?tag=v0.1.0
```

## PyPI Yayini

```bash
cd "/Users/baris/Documents/GitHub/Tilavet/tilavet-aligner"
rm -rf dist build src/*.egg-info
python3 -m build
python3 -m twine check dist/*

# Test PyPI'de smoke (onerilir):
python3 -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    tilavet-aligner==0.1.0
tilavet-align --help

# Production PyPI:
python3 -m twine upload dist/*
```

`tilavet-phonemizer` icin kullandigin ayni PyPI API token'i (veya
`__token__` username + token) `~/.pypirc`'tan otomatik kullanilacak.

## Yayindan Sonra Smoke Test

```bash
python3 -m venv /tmp/check_aligner && source /tmp/check_aligner/bin/activate
pip install --upgrade tilavet-phonemizer tilavet-aligner
python3 - <<'PY'
from tilavet_phonemizer import Phonemizer
from tilavet_aligner.contract import parse_target

target = Phonemizer().phonemize("بِسْمِ ٱللَّهِ").to_alignment_dict()
parsed = parse_target(target)
print("symbols:", parsed.symbols)
print("words:", [w.token for w in parsed.words])
PY
```

`tilavet-align` komutu da PATH'te olmali.

## Backwards Compat Cangili

- Bu paket `tilavet-phonemizer >= 0.2.0`'in `to_alignment_dict()` cikitisina
  bagli. 0.1.x serisinde `to_alignment_dict` yoktu — `pyproject.toml`'da
  ekstra `phonemizer` `>=0.2.0` shartiyla isaretledim.
- PAUSE metadata kontratiyla uyumlu. CTC class listesinde PAUSE yoktur;
  bunu degistirmeyin (yoksa hem phonemizer hem aligner'a yeni minor versiyon
  cikarmamiz gerek).

## Sonraki Adimlar (Yayin Sonrasi)

1. **Wav2Vec2 smoke test** — `facebook/wav2vec2-xlsr-arabic-960h` ile
   gercek bir Fatiha kaydi uzerinde alignment dogrulugu olc. Bagimsiz repo:
   `Tilavet/tilavet-aligner/examples/wav2vec2_real_audio.py` (henuz yok).
2. **iOS swift port parity** — `Tilavet/Aligner/TilavetAligner.swift`
   icinde Python ile parity tests; Cmd-U ile gec.
3. **CTC model entegrasyonu** — `tilavet-acoustic` ile birlikte; egitilen
   model `TilavetCTC.mlmodelc` olarak iOS bundle'a girer; Aligner Swift
   tarafinda gercek posteriors tuketmeye baslar.

## Kontroller / Pre-Publish Checklist

```
- [ ] git status temiz, tum degisiklikler commit
- [ ] CHANGELOG.md [0.1.0] tarih: 2026-05-15
- [ ] pyproject.toml version = "0.1.0"
- [ ] LICENSE dosyasi mevcut (MIT)
- [ ] README.md guncel
- [ ] python -m build basariyla calisti (15 KB whl, 17 KB sdist)
- [ ] twine check dist/* PASSED
- [ ] pytest 35 passed, 1 skipped, 0 failed
- [ ] PyPI token erisilebilir
- [ ] GitHub'da `tialkan/tilavet-aligner` repo ac (henuz yoksa)
```
