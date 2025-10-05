import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


st.set_page_config(
    page_title="Spiral Visualization",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Spiral Visualization</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Interactive Mathematical Spiral Generator</p>", unsafe_allow_html=True)

st.markdown("---")

with st.expander("About this visualization", expanded=False):
    st.markdown("""
    This interactive visualization demonstrates a parametric spiral using Altair charts.
    
    - **Number of points**: Controls the density of points along the spiral
    - **Number of turns**: Controls how many complete rotations the spiral makes
    
    The spiral is generated using polar coordinates converted to Cartesian:
    - `θ = 2π × turns × t` (where t goes from 0 to 1)
    - `r = t` (radius increases linearly)
    - `x = r × cos(θ)`
    - `y = r × sin(θ)`
    """)

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Controls")
    
    num_points = st.slider(
        "Number of points in spiral",
        min_value=1,
        max_value=10000,
        value=1100,
        step=100
    )
    
    num_turns = st.slider(
        "Number of turns in spiral",
        min_value=1,
        max_value=300,
        value=31,
        step=1
    )
    
    st.markdown("---")
    
    st.markdown("### Statistics")
    st.metric("Total Points", f"{num_points:,}")
    st.metric("Total Turns", num_turns)
    st.metric("Points per Turn", f"{num_points // num_turns if num_turns > 0 else 0}")

with col2:
    st.markdown("### Spiral Visualization")
    
    indices = np.linspace(0, 1, num_points)
    theta = 2 * np.pi * num_turns * indices
    radius = indices
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "idx": indices,
        "rand": np.random.randn(num_points),
    })
    
    chart = alt.Chart(df, height=700, width=700).mark_point(filled=True).encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    )
    
    st.altair_chart(chart, use_container_width=True)

st.markdown("---")

with st.expander("View Data Table"):
    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=300
    )
    
    csv_output = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Spiral Data CSV",
        csv_output,
        "spiral_data.csv",
        "text/csv",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Spiral Visualization | Built with Streamlit & Altair</p>",
    unsafe_allow_html=True
)