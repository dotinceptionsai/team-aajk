import streamlit as st

st.set_page_config(
    page_title="Fantastic Filter",
    page_icon="👋",
    layout="wide",
)

st.session_state.update(st.session_state)

st.write("# Fantastic Filter🪄 Setup Wizard")

st.markdown(
    """
    The Fantastic Filter🪄is an application that helps Call-Centers to filter live-conversations between an employee and a customer. It tells
    which sentences are relevant to their domain (their FAQs) and which are likely not. 
    
    In general, Call-Centers already own a question-answering software that searches their FAQs or Knowledge Base. However, those systems are slow and cannot afford to run on each and every utterance spoken by the customer or the employee. It is here that the Fantastic Filter comes in:
    it is a fast pre-filtering on the conversation that keeps only sentences that are likely to be relevant to the downstream question-answering system.
    
    This backoffice application will guide you through the setup of the Fantastic Filter for your Call-Center.
    
    For this demo app, we only allow a pre-defined set of FAQs that are listed below. Each of those have been scraped from the internet. In the future, we will allow you to upload your own knowledge base.
    
    **👈 Start setup process by clicking on the left menu!**
"""
)
