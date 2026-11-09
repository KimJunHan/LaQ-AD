# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 교사 실행 래퍼용 "common_v2 자리 바꾸기". answer_grammar.py · laq_cache_layout.py · extract_teacher_cache_v2.py 는
#   `import common_v2 as V` 로 정답 문법(클래스 표·파서·형식 함수)을 읽는다. B2D 정답은 줄·필드·토큰 수가 같고(547) 클래스 표·
#   필드 정의만 다르므로, 그 파일들을 고치지 않고 이 모듈로 sys.modules["common_v2"] 를 common_b2d 로 바꾼 뒤 import 한다.
#   반드시 그 파일들을 import 하기 "전에" install() 을 불러야 한다(이미 import 됐으면 실패시킨다 — 조용한 혼용 금지).
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_DEPENDENTS = ("answer_grammar", "laq_cache_layout", "extract_teacher_cache_v2", "teacher_metrics", "eval_teacher")


def install(action_k):
    import common_b2d as B
    loaded = [m for m in _DEPENDENTS if m in sys.modules and getattr(sys.modules[m], "V", None) is not None
              and getattr(sys.modules[m], "V") is not B]
    if loaded:
        raise RuntimeError(f"b2d_teacher_shim.install 전에 이미 nuScenes common_v2 로 import 된 모듈: {loaded}")
    B.set_action_k(action_k)
    sys.modules["common_v2"] = B
    return B


def tokenizer_with_b2d_tokens(path, special_tokens):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    n = tok.add_tokens(list(special_tokens), special_tokens=True)
    if n != len(special_tokens):
        raise RuntimeError(f"특수토큰 추가 {n} != {len(special_tokens)}")
    return tok
