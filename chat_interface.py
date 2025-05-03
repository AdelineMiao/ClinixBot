# chat_interface.py
import streamlit as st
import time
import random
from translations import translations

class ChatInterface:
    def __init__(self, rag_model, lang="zh"):
        self.rag_model = rag_model
        self.lang = lang
        self.t = translations
    
    def _get_avatar(self, is_user):
        """Get avatar icon"""
        return "👤" if is_user else "🏥"
    
    def _add_message(self, message, is_user=True):
        """Add message to chat history"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append({
            "message": message,
            "is_user": is_user,
            "timestamp": time.time(),
            "lang": self.lang  # Save message language
        })
    
    def _get_initial_greeting(self):
        """Get initial welcome message based on current language"""
        greetings = [
            self.t.chat[self.lang]["initial_greeting1"],
            self.t.chat[self.lang]["initial_greeting2"],
            self.t.chat[self.lang]["initial_greeting3"]
        ]
        return random.choice(greetings)
    
    def _handle_user_input(self):
        """Process user input"""
        user_input = st.session_state.user_input
        
        if user_input:
            # Add user message
            self._add_message(user_input, is_user=True)
            
            # Clear input box
            st.session_state.user_input = ""
            
            # Get diagnosis result
            with st.spinner(self.t.chat[self.lang]["analyzing"]):
                diagnosis_result = self.rag_model.get_diagnosis(user_input, lang=self.lang)
            
            # Save current diagnosis
            st.session_state.current_diagnosis = diagnosis_result["diagnosis"]
            
            # Add bot response
            self._add_message(diagnosis_result["diagnosis"], is_user=False)
            
            # If diagnosis contains a result, get medication recommendations
            diagnosis_key = "初步诊断" if self.lang == "zh" else "Preliminary diagnosis"
            if diagnosis_key in diagnosis_result["diagnosis"]:
                with st.spinner(self.t.chat[self.lang]["generating_recommendations"]):
                    medications = self.rag_model.get_medication_recommendations(diagnosis_result["diagnosis"], lang=self.lang)
                    st.session_state.recommended_medications = medications
    
    def render(self):
        """Render chat interface"""
        t = self.t.chat[self.lang]
        ui = self.t.ui[self.lang]
        
        # Initialize chat history if needed
        if 'chat_history' not in st.session_state or not st.session_state.chat_history:
            self._add_message(self._get_initial_greeting(), is_user=False)
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(name="user" if message["is_user"] else "assistant", avatar=self._get_avatar(message["is_user"])):
                    st.write(message["message"])
        
        # User input
        st.text_input(t["input_placeholder"], key="user_input", on_change=self._handle_user_input)
        
        # If medication recommendations exist, display them with pharmacy/hospital buttons
        if 'recommended_medications' in st.session_state and st.session_state.recommended_medications:
            with st.expander(t["view_recommendations"], expanded=True):
                st.markdown(st.session_state.recommended_medications)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(t["find_nearby_pharmacy"]):
                        st.session_state.active_view = "pharmacy"
                        st.experimental_rerun()
                with col2:
                    if st.button(t["find_nearby_hospital"]):
                        st.session_state.active_view = "hospital"
                        st.experimental_rerun()