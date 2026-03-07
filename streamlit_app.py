"""
VisiQ — Streamlit wrapper
Embeds the self-contained VisiQ HTML/JS app directly in Streamlit.

Deploy on Streamlit Cloud:
  1. Put this file (streamlit_app.py) and visiq_app.html in the same repo root.
  2. Add requirements.txt with just: streamlit
  3. Deploy — no Flask, no numpy, no plotly needed.
"""

import streamlit as st
import pathlib

st.set_page_config(
    page_title="VisiQ — AI Data Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Remove all Streamlit chrome so the embedded app fills the viewport cleanly
st.markdown(
    """
    <style>
      /* Hide Streamlit header, footer, and main block padding */
      #MainMenu, header, footer { visibility: hidden; height: 0; }
      .block-container { padding: 0 !important; max-width: 100% !important; }
      iframe { border: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load the HTML file ────────────────────────────────────────────────────────
HTML_FILE = pathlib.Path(__file__).parent / "visiq_app.html"

if not HTML_FILE.exists():
    st.error(
        "❌ `visiq_app.html` not found next to `streamlit_app.py`.\n\n"
        "Make sure both files are in the **same folder** in your repository."
    )
    st.stop()

html_content = HTML_FILE.read_text(encoding="utf-8")

# ── Embed via st.components ───────────────────────────────────────────────────
# height is set tall so all content is visible; scrolling handled inside iframe
st.components.v1.html(html_content, height=1100, scrolling=True)
