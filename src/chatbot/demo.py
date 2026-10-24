from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Tiki Shopping Assistant", page_icon="🛒")
st.title("Tiki Shopping Assistant")
st.caption("Hỏi bất cứ điều gì về sản phẩm trên Tiki")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ví dụ: cốc giữ nhiệt chất lượng tốt dưới 300k"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            resp = requests.post(API_URL, json={"query": query})
            data = resp.json()
            answer = data["answer"]
            sources = data["sources"]

        st.write(answer)

        with st.expander("Nguồn tham khảo"):
            for s in sources:
                st.write(f"- **{s['name']}** ({s['doc_type']}, score: {s['score']})")

    st.session_state.messages.append({"role": "assistant", "content": answer})
