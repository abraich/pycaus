import streamlit as st

# Home
def app():
    st.title("ClassCaus & SurvCaus")

    st.markdown(
        """
        This multi-page web app demonstrates the capabilities of our models vis a vis the state of the art.
        """
    )
    # papers:
    st.markdown(
        """ 
        [ClassCaus paper](https://arxiv.org/abs/1908.04346)
        """
    )
    st.markdown(
        """ 
        [SurvCaus paper](https://arxiv.org/abs/1907.09091)
        """
    )
    