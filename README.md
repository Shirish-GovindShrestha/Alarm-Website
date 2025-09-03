
AI-Powered Chatbot with Alarm Functionality

This project is an AI-powered chatbot built with Flask, TinyBERT (for intent classification and Named Entity Recognition), and a Finite State Machine (FSM) for managing conversational flow.
It supports slot filling, context-aware conversations, and a smart alarm system with validation for time (AM/PM), dates, and repeat scheduling.


Features

* NLP-Powered Chatbot – Uses TinyBERT for both intent recognition and entity extraction
* Finite State Machine (FSM) – Handles context-aware dialogue and manages slot-filling (time, date, label, repeat)
* Alarm Manager – Users can set alarms with labels, repetition (daily/none), and get notified when alarms trigger
* Validation – Built-in FSM checks for valid times (AM/PM, 24h), valid dates, and ensures all required slots are filled
* Web Interface – Frontend built with HTML, CSS, and JavaScript for interactive chat
* SQLite Database – Stores and manages chatbot-related data persistently


Tech Stack

* Backend: Flask (Python)
* NLP Models: TinyBERT (Transformers)
* FSM & Dialogue Management: Custom fsm.py
* Database: SQLite
* Frontend: HTML, CSS, JavaScript (chat UI in templates/index.html)
* Alarm Management: Custom alarm\_manager.py with datetime


Project Structure


chatbot/
│── app.py                  # Main Flask app
│── alarm_manager.py        # Alarm Manager (add/check alarms)
│── models/
│   ├── intent_model.py     # TinyBERT intent classifier
│   ├── ner_model.py        # TinyBERT NER model
│   └── fsm.py              # Finite State Machine for slot filling & context
│── templates/
│   └── index.html          # Frontend chatbot UI
│── static/                 # (Optional: JS/CSS files for UI)
│── requirements.txt        # Dependencies


How It Works

1. User Input → The chatbot frontend sends a message to Flask
2. Intent & NER → TinyBERT extracts the user’s intent and entities (time, date, etc.)
3. FSM Processing → The FSM manages conversation context, slot filling, and validation
4. Alarm Setting → If the intent is to set an alarm, details are passed to the AlarmManager
5. Alarm Check → Flask exposes /check-alarms for frontend polling to trigger alarms in real-time
6. Response → The chatbot responds with confirmations, queries for missing info, or alarm notifications
