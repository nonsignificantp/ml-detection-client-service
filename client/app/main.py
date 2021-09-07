import streamlit as st
from src.app import front, flow
      

@flow()
def app():
    image = front.inference()
    st.image(image)

if __name__ == "__main__":
    app()
