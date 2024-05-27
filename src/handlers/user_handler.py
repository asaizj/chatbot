import streamlit as st

# Manejar la entrada de datos del usuario
def handle_userinput(user_question):
    # Se pasa la pregunta del usuario a la cadena de conversacion
    response = st.session_state.conversation({'question': user_question})
    # Se actualiza el historial de chat almacenado en la sesion con la respuesta generada por el chatbot
    st.session_state.chat_history = response['chat_history']
    # Se itera sobre cada mensaje en el historial de chat. Si el indice del mensaje es par, se trata de un mensaje del usuario y se muestra por pantalla
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            #st.write(user_template.replace("{{MSG}}", message.content, is_user = True, key = str(i)), unsafe_allow_html = True)
            st.write(message.content, is_user = True, key = str(i))
        # Si el indice es impar, asume que se trata de una respuesta del chatbot
        else:
            #st.write(bot_template.replace("{{MSG}}", message.content, key = str(i)), unsafe_allow_html = True)
            st.write(message.content, key = str(i))