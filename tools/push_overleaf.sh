#!/bin/bash
# 수정(2026-09-29 v2): IV 논문 Overleaf 동기화.
#   Overleaf 프로젝트는 자체 git 저장소이므로 LaQ-AD 히스토리를 밀어넣지 않고,
#   프로젝트를 클론해 논문 파일만 갱신·커밋·푸시한다. 덮어쓰기 전 diff 를 반드시 보여준다.
#
#   사용법(토큰이 대화/파일에 남지 않도록 사용자 터미널에서 실행):
#     ! OVERLEAF_TOKEN=olp_xxxxx bash tools/push_overleaf.sh
#   선택: 다른 프로젝트면  ! OVERLEAF_TOKEN=... bash tools/push_overleaf.sh <project-id>
#   토큰 발급: Overleaf > Account Settings > Git Integration (Premium 기능).
set -eu
PROJ=${1:-6a8d686fc049644c93f37675}
: "${OVERLEAF_TOKEN:?OVERLEAF_TOKEN 환경변수가 필요합니다 (예: OVERLEAF_TOKEN=olp_xxx bash $0)}"
SRC=/home1/irteam/data-vol1/src/HiP-AD/docs
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
RED() { sed "s/${OVERLEAF_TOKEN}/<REDACTED>/g"; }

echo "[1/5] 프로젝트 클론 ($PROJ)"
git clone "https://git:${OVERLEAF_TOKEN}@git.overleaf.com/${PROJ}" "$WORK/ol" 2>&1 | RED | tail -2
cd "$WORK/ol"
echo "[2/5] 현재 Overleaf 내용"
git log --oneline -3 || true
ls -1

echo "[3/5] 논문 파일 반영"
# 본문: Overleaf 쪽 파일명이 main.tex 이면 그쪽에, 아니면 iv_main_ko.tex 로 쓴다.
if [ -f main.tex ] && [ ! -f iv_main_ko.tex ]; then TGT=main.tex; else TGT=iv_main_ko.tex; fi
cp "$SRC/iv/iv_main_ko.tex" "$TGT"
cp "$SRC/references.bib" references.bib
rm -rf figures && mkdir -p figures && cp "$SRC"/iv/figures/*.pdf figures/   # 수정(2026-09-29): 삭제된 그림이 남지 않도록 동기화
echo "  본문 -> $TGT, references.bib, figures/*.pdf"

echo "[4/5] 변경 요약 (덮어쓰기 전 확인)"
git add -A
git diff --cached --stat | tail -12
if git diff --cached --quiet; then echo "변경 없음 — 이미 최신. 종료"; exit 0; fi

echo "[5/5] 커밋·푸시"
git -c user.name="Junhan Kim" -c user.email="kimjunhan1605@gmail.com" \
    commit -q -m "${OVERLEAF_MSG:-Sync IV draft ($(date +%Y-%m-%d))}"  # 수정(2026-10-12): 고정 문구(896x448)가 현 모델과 달라 메시지를 인자로
git push 2>&1 | RED | tail -3
echo "완료 — Overleaf 에서 xelatex 컴파일 후 6쪽 유지 여부를 확인하세요."
