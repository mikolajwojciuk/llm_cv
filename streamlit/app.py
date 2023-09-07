import streamlit as st
from src.interactive_cv import InteractiveCv


cv = InteractiveCv()
cv.add_pdf_file("streamlit/resources/cv.pdf")
cv.add_csv_file("streamlit/resources/cv.csv")
cv.init_chain()

st.title("Interactive Cv")
st.header(
    "Hi! This app is created to allow You to ask questions about Mikołaj! Give it a try!"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about Mikołaj!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = cv(prompt)
        full_response += response
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
