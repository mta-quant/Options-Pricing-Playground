import streamlit as st

# Title
st.title("Hello World Streamlit App")

# Text
st.write("This is a test deployment to Streamlit Cloud!")

# Input box example
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! 👋")

# Button example
if st.button("Click me"):
    st.write("You clicked the button!")
