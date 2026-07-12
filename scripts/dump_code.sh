#!/usr/bin/env bash
set -euo pipefail
BASE="/home/sehaxe/aria"
OUT_DEFAULT="aria-code.md"
OUT="${1:-$OUT_DEFAULT}"
# If OUT is relative, place it under BASE
[[ "$OUT" != /* ]] && OUT="$BASE/$OUT"
SRC="$BASE/src"
CFG="$BASE/configs"
TST="$BASE/tests"
SCR="$BASE/scripts"

{
  echo "# Aria — Full Source"
  echo
  src_lines=$(find "$SRC" -name '*.py' | xargs wc -l | tail -1 | awk '{print $1}')
  src_files=$(find "$SRC" -name '*.py' | wc -l)
  tst_lines=$(find "$TST" -name '*.py' 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
  tst_files=$(find "$TST" -name '*.py' 2>/dev/null | wc -l)
  echo "Total: $src_lines Python lines across $src_files files (src), ${tst_lines:-0} in ${tst_files:-0} test files"
  echo

  # src + pyx
  while IFS= read -r f; do
    rel="${f#$BASE/}"
    ext="${f##*.}"
    echo "---"
    echo "### $rel"
    echo '```'"$ext"''
    cat "$f"
    echo '```'
    echo
  done < <(find "$SRC" -name '*.py' -o -name '*.pyx' | sort)

  # configs
  while IFS= read -r f; do
    rel="${f#$BASE/}"
    echo "---"
    echo "### $rel"
    echo '```yaml'
    cat "$f"
    echo '```'
    echo
  done < <(find "$CFG" -name '*.yaml' -o -name '*.yml' 2>/dev/null | sort)

  # tests
  while IFS= read -r f; do
    rel="${f#$BASE/}"
    echo "---"
    echo "### $rel"
    echo '```python'
    cat "$f"
    echo '```'
    echo
  done < <(find "$TST" -name '*.py' 2>/dev/null | sort)

  # root-level scripts
  for f in \
    "$BASE/prepare_data.py" \
    "$BASE/pyproject.toml" \
    "$BASE/.python-version"; do
    [ -f "$f" ] || continue
    rel="${f#$BASE/}"
    ext="${f##*.}"; [ "$ext" = "toml" ] && ext="toml" || true
    case "$rel" in
      *.toml)  lang="toml" ;;
      *.py)    lang="python" ;;
      *.yaml|*.yml) lang="yaml" ;;
      .python-version) lang="text" ;;
      *)       lang="text" ;;
    esac
    echo "---"
    echo "### $rel"
    echo '```'"$lang"''
    cat "$f"
    echo '```'
    echo
  done

  # AGENTS.md
  for f in "$BASE/AGENTS.md" "$BASE/CLAUDE.md"; do
    [ -f "$f" ] || continue
    rel="${f#$BASE/}"
    echo "---"
    echo "### $rel"
    echo '```markdown'
    cat "$f"
    echo '```'
    echo
  done

} > "$OUT"
echo "Done: $(wc -l < "$OUT") lines → $OUT"
