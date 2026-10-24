"""Run: python -m runtime.chatbot.demo_chat --query "sản phẩm chất lượng tốt".

Omit --query for an interactive session. --context-only does not call an LLM.
"""
from __future__ import annotations

import argparse

from . import llm


def run_turn(rag, query, history, *, top_k=5, stream=False, context_only=False):
    search_query = query if context_only else llm.rewrite_query(query, history)
    docs = rag.search(search_query, top_k=top_k)
    if context_only:
        print(llm.build_context(docs) or llm.NO_DATA)
        return
    if stream:
        pieces = []
        try:
            for piece in llm.ask_stream(query, docs, history):
                print(piece, end="", flush=True)
                pieces.append(piece)
        finally:
            print()
        answer = "".join(pieces)
    else:
        answer = llm.ask(query, docs, history)
        print(answer)
    if docs:
        print("\nNguồn truy xuất:")
        for i, doc in enumerate(docs, 1):
            metadata = doc.get("metadata") or {}
            print(f"[{i}] {metadata.get('name') or doc['id']} | {doc['id']}")
            if metadata.get("product_url"):
                print(f"    {metadata['product_url']}")
    # Interrupted streams never reach this point, so partial responses are not remembered.
    history.extend([{"role": "user", "content": query}, {"role": "assistant", "content": answer}])
    history[:] = history[-8:]


def main():
    parser = argparse.ArgumentParser(description="Chat with the indexed Tiki knowledge base")
    parser.add_argument("--query", help="Ask one question; omit for interactive mode")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--context-only", action="store_true", help="Retrieve context without calling any LLM API")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 8:
        parser.error("top-k must be between 1 and 8")
    if args.query is not None and not args.query.strip():
        parser.error("query must not be blank")
    if not args.context_only:
        try:
            llm.configured_providers()
        except llm.LLMConfigurationError as exc:
            parser.error(str(exc))
    from .retrieval import TikiRAG
    rag = None
    try:
        rag = TikiRAG()
        history = []
        if args.query is not None:
            run_turn(rag, args.query, history, top_k=args.top_k,
                     stream=args.stream, context_only=args.context_only)
        else:
            print("Nhập câu hỏi. Gõ /clear để xóa lịch sử; /exit để thoát.")
            while True:
                query = input("Bạn: ").strip()
                if query.lower() in ("/exit", "exit", "quit"):
                    break
                if query.lower() == "/clear":
                    history.clear()
                    print("Đã xóa lịch sử.")
                    continue
                if not query:
                    continue
                try:
                    run_turn(rag, query, history, top_k=args.top_k,
                             stream=args.stream, context_only=args.context_only)
                except (llm.LLMError, ValueError) as exc:
                    print(f"Không hoàn tất câu trả lời: {exc}")
    except (EOFError, KeyboardInterrupt):
        print("\nĐã kết thúc phiên.")
    except (llm.LLMError, ValueError) as exc:
        parser.exit(1, f"Không hoàn tất câu trả lời: {exc}\n")
    finally:
        if rag is not None:
            rag.close()
        llm.close_clients()


if __name__ == "__main__":
    main()
