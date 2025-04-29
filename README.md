# ClinixBot: Intelligent Medical Assistant

![ClinixBot Logo](https://via.placeholder.com/150x150.png?text=ClinixBot)

## Overview

ClinixBot is an intelligent medical assistant web application designed to provide preliminary medical diagnoses and healthcare recommendations based on user-described symptoms. The application leverages AI technology to analyze symptoms, offer medication recommendations, locate nearby medical facilities, and visualize healthcare data. The app fully supports both English and Chinese languages with an easy language switch feature.

## Features

- **Language Switching**: Seamlessly switch between English and Chinese throughout the entire application
- **Chat-based Diagnosis**: AI-powered symptom analysis and preliminary diagnosis in your preferred language
- **Medication Recommendations**: Suggestions for over-the-counter medications based on diagnosis
- **Pharmacy Finder**: Locate nearby pharmacies that carry recommended medications
- **Hospital Finder**: Find nearby hospitals and filter by department and availability
- **Medical Data Analysis**: Visualize and analyze healthcare data through interactive dashboards

## Tech Stack

- **Frontend**: Streamlit (Python web application framework)
- **Backend**: Python 3.8+
- **AI/ML**: 
  - LangChain for RAG (Retrieval-Augmented Generation)
  - OpenAI GPT-4 for natural language processing
  - FAISS for vector storage and similarity search
- **Data Storage**: CSV files for development, vector stores for embeddings
- **APIs**: OpenAI API for AI capabilities
- **Multilingual Support**: Centralized translation dictionary system

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AdelineMiao/ClinixBot.git
   cd ClinixBot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Project Structure

```
ClinixBot/
├── app.py                  # Main application entry point
├── translations.py         # Central translation dictionary for multilingual support
├── components/             # UI components
│   ├── chat_interface.py   # Chat diagnosis interface
│   ├── hospital_finder.py  # Hospital finder component
│   ├── pharmacy_finder.py  # Pharmacy finder component
│   └── visualization_dashboard.py  # Data visualization component
├── models/
│   └── rag_model.py        # RAG model for medical diagnosis (multilingual capable)
├── utils/
│   └── data_processor.py   # Data processing utilities
├── data/
│   └── hospital_records_2021_2024_with_bills.csv  # Sample medical data
├── vector_store/           # FAISS vector store for embeddings
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser at `http://localhost:8501`

3. Select your preferred language (Chinese or English) using the dropdown in the sidebar

4. Use the sidebar to navigate between different features:
   - **Chat Diagnosis**: Describe your symptoms to get a preliminary diagnosis
   - **Medical Data Analysis**: Explore healthcare data visualizations
   - **Find Pharmacy**: Locate nearby pharmacies that carry recommended medications
   - **Find Hospital**: Find nearby hospitals filtered by department and availability

## Language System

The application uses a centralized translation system managed through the `translations.py` file:

- All text content is stored in a structured dictionary with language-specific sub-dictionaries
- Each component accesses translations based on the currently selected language
- Language selection is stored in Streamlit's session state
- The AI models automatically respond in the selected language

### Adding New Languages

To add a new language:

1. Edit the `translations.py` file
2. Add the new language code and translations to the appropriate dictionaries
3. Update the language selector in the sidebar to include the new option
4. Enhance the RAG model to support responses in the new language

## Extending the RAG Model

The RAG model supports multilingual functionality with separate prompt templates for each language. To improve the diagnosis capabilities:

1. Add more medical data to the CSV files in the `data` directory
2. Modify the `MedicalRAGModel` class in `models/rag_model.py` to enhance the retrieval and generation components
3. Add language-specific prompt templates for any new languages

## Security Note

This application uses FAISS for vector store serialization. By default, it rebuilds the vector store from source data to avoid security issues with pickle deserialization.

## Disclaimer

ClinixBot provides preliminary diagnostic references only and cannot replace professional medical diagnosis and treatment advice. For serious symptoms, please seek immediate medical attention.

## License

[Your License Here]

## Contributors

- Adeline Miao
- Tianlun Li
- Nancy Xu
- Qixuan Zhang

## Acknowledgements

- Streamlit for the amazing web application framework
- OpenAI for providing the GPT-4 API
- LangChain for the RAG implementation framework
