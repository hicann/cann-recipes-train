# 安装 Wordle 游戏环境 + vf-eval（NLTK 语料库已由 02.03 准备）
set -euo pipefail

PRIMERL_DIR=./prime-rl
if [ -d "$PRIMERL_DIR/.git" ]; then
  echo "Using existing prime-rl at: $PRIMERL_DIR"
else
  git clone https://gitcode.com/gh_mirrors/pr/prime-rl.git "$PRIMERL_DIR"
fi
cd "$PRIMERL_DIR"
git checkout 188192ce64b2b7acf82e83ae36cfb0632bebde5b

VERIFIERS_REV=d822f6aca7a967fc6698b1d595524c6278d84a5c

if [ ! -e deps/verifiers/.git ] ||
    [ "$(git -C deps/verifiers rev-parse HEAD 2>/dev/null || true)" != "$VERIFIERS_REV" ]; then
  rm -rf deps/verifiers
  git init deps/verifiers
  git -C deps/verifiers remote add origin \
    https://gitcode.com/GitHub_Trending/ver/verifiers.git
  git -C deps/verifiers fetch --depth=1 origin "$VERIFIERS_REV"
  git -C deps/verifiers checkout --detach FETCH_HEAD
fi

export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p "$UV_CACHE_DIR"

# Prime-RL Wordle verifier 需要自己的 venv 环境。
uv venv .venv-wordle-legacy --clear
source .venv-wordle-legacy/bin/activate

uv pip install -e deps/verifiers --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
uv pip install -e deps/verifiers/environments/wordle --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

echo '=== 验证 ==='
echo "vf-eval: $(which vf-eval)"
echo "prime-rl revision: $(git rev-parse HEAD)"
echo "verifiers revision: $(git -C deps/verifiers rev-parse HEAD)"
