from flask import Flask, request, jsonify, render_template, session
from models.fsm import ConversationalAlarmFSM
from alarm_manager import AlarmManager
import uuid
import json
from datetime import datetime
import re

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Store FSM instances per session
fsm_sessions = {}

# Persistent alarm manager (in-memory)
alarm_manager = AlarmManager()

import whisper
import tempfile

# Load model once at startup (this is expensive)
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_file.save(tmp.name)
        
        try:
            result = model.transcribe(tmp.name)
            return jsonify({'transcript': result['text']})
        finally:
            os.unlink(tmp.name)


def get_or_create_fsm():
    """Get or create FSM instance for current session"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    if session_id not in fsm_sessions:
        fsm_sessions[session_id] = ConversationalAlarmFSM()
    
    return fsm_sessions[session_id]

def parse_sql_for_display(sql_query, entities):
    """Parse SQL to extract alarm information for display.
    Prefer extracting from SQL; fallback to supplied entities.
    """
    if not sql_query or sql_query.startswith('--'):
        return None

    label = None
    time_val = None
    date_val = None
    repeat_val = None

    # INSERT INTO alarms (col1, col2, ...) VALUES ('v1','v2',...);
    m = re.search(r"insert\s+into\s+alarms\s*\(([^\)]+)\)\s*values\s*\(([^\)]+)\)", sql_query, re.IGNORECASE)
    if m:
        cols = [c.strip().strip('`"') for c in m.group(1).split(',')]
        raw_vals = m.group(2)
        # split on commas not inside quotes
        parts = re.findall(r"'(?:[^']|''*)*'|[^,]+", raw_vals)
        vals = []
        for p in parts:
            v = p.strip()
            if v.startswith("'") and v.endswith("'"):
                v = v[1:-1].replace("''", "'")
            vals.append(v)
        mapping = {c: (vals[i] if i < len(vals) else '') for i, c in enumerate(cols)}
        label = mapping.get('label')
        time_val = mapping.get('time')
        date_val = mapping.get('date')
        repeat_val = mapping.get('repeat') or mapping.get('repeat_pattern')

    # Fallbacks from entities if any field missing
    entities = entities or {}
    if not label:
        label = entities.get('label', 'Unnamed Alarm')
    if not time_val:
        time_val = entities.get('time', 'Not specified')
    if not date_val:
        date_val = entities.get('date', datetime.now().strftime('%Y-%m-%d'))
    if not repeat_val:
        repeat_val = entities.get('repeat', 'None')

    alarm_info = {
        'id': str(uuid.uuid4())[:8],
        'label': label,
        'time': time_val,
        'date': date_val,
        'repeat': repeat_val,
        'status': 'active' if 'insert into' in sql_query.lower() else 'updated',
        'created_at': datetime.now().strftime('%H:%M:%S')
    }
    
    return alarm_info

@app.route('/')
def home():
    # Initialize new session
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({
                "response": "I didn't receive your message. Could you try again?",
                "state": "error",
                "sql_query": "",
                "alarm_data": None,
                "conversation_stats": {}
            })

        user_input = data["message"]
        fsm = get_or_create_fsm()
        
        # Snapshot entities before processing (FSM may reset after confirm)
        prev_entities = fsm.entities.copy()
        
        # Process input through FSM
        response = fsm.process_input(user_input)
        
        # Extract SQL query if generated
        sql_query = ""
        alarm_data = None
        
        # Check if action was executed (SQL generated)
        if "SQL:" in response:
            marker = "💾 SQL:"
            sql_start = response.find(marker)
            if sql_start != -1:
                sql_start += len(marker)
                sql_end = response.find("\n\n", sql_start)
                if sql_end == -1:
                    sql_end = len(response)
                sql_query = response[sql_start:sql_end].strip()
            
            # Clean up response (remove SQL from display)
            response = response[:response.find("💾 SQL:")].strip()
            if response.endswith("\n"):
                response = response[:-1]
            
            # Prefer prev_entities if current were cleared by reset
            entities_for_alarm = fsm.entities if fsm.entities else prev_entities
            
            # Parse alarm data for frontend display
            alarm_data = parse_sql_for_display(sql_query, entities_for_alarm)

            # If a new alarm was set/updated/cancelled, sync with the AlarmManager (DB)
            try:
                if fsm.current_intent == 'set_alarm' or (sql_query.lower().startswith('insert into alarms')):
                    # Always trust entities captured during conversation for DB storage
                    label_val = entities_for_alarm.get('label', 'Alarm')
                    time_val = entities_for_alarm.get('time', '')
                    date_val = entities_for_alarm.get('date', datetime.now().strftime('%Y-%m-%d'))
                    repeat_val = entities_for_alarm.get('repeat', 'None')

                    alarm_manager.add_alarm(
                        time_str=time_val,
                        date_str=date_val,
                        repeat_str=repeat_val,
                        label=label_val
                    )
                elif fsm.current_intent == 'update_alarm' or sql_query.lower().startswith('update alarms'):
                    label_val = entities_for_alarm.get('label')
                    if label_val:
                        alarm_manager.update_alarms(
                            label=label_val,
                            new_time=entities_for_alarm.get('time'),
                            new_date=entities_for_alarm.get('date'),
                            new_repeat=entities_for_alarm.get('repeat')
                        )
                elif fsm.current_intent == 'cancel_alarm' or sql_query.lower().startswith('delete from alarms'):
                    label_val = entities_for_alarm.get('label')
                    if label_val:
                        alarm_manager.delete_alarms(
                            label=label_val,
                            time_str=entities_for_alarm.get('time'),
                            date_str=entities_for_alarm.get('date')
                        )
            except Exception:
                # Non-fatal: continue without blocking chat response
                pass

        # Get conversation statistics
        stats = fsm.get_conversation_stats()
        
        # Ensure entities are shown on the UI right after creating an alarm
        entities_payload = fsm.entities
        missing_fields_payload = fsm.missing_fields
        if alarm_data and (not entities_payload or len(entities_payload) == 0):
            entities_payload = entities_for_alarm
            missing_fields_payload = []
        
        return jsonify({
            "response": response,
            "state": fsm.state,
            "current_intent": fsm.current_intent,
            "entities": entities_payload,
            "missing_fields": missing_fields_payload,
            "sql_query": sql_query,
            "alarm_data": alarm_data,
            "conversation_stats": stats,
            "user_name": fsm.user_name
        })
        
    except Exception as e:
        return jsonify({
            "response": f"Sorry, I encountered an error: {str(e)}. Let's start fresh!",
            "state": "error",
            "sql_query": "",
            "alarm_data": None,
            "conversation_stats": {}
        }), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation state"""
    try:
        fsm = get_or_create_fsm()
        fsm.reset()
        
        return jsonify({
            "response": fsm.get_random_response("greetings"),
            "state": "IDLE",
            "message": "Conversation reset successfully!"
        })
    except Exception as e:
        return jsonify({
            "response": "Hi! I'm your alarm assistant. How can I help you today?",
            "state": "IDLE",
            "message": f"Reset with error: {str(e)}"
        })

@app.route('/help', methods=['GET'])
def get_help():
    """Get help information"""
    fsm = get_or_create_fsm()
    help_text = fsm.show_help()
    
    return jsonify({
        "response": help_text,
        "state": "help"
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get conversation statistics"""
    try:
        fsm = get_or_create_fsm()
        stats = fsm.get_conversation_stats()
        conversation_log = fsm.export_conversation_log()
        
        return jsonify({
            "stats": stats,
            "log": conversation_log
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "stats": {},
            "log": {}
        })

@app.route('/demo', methods=['GET'])
def run_demo():
    """Run a demo conversation"""
    fsm = ConversationalAlarmFSM()  # Create fresh instance for demo
    
    demo_inputs = [
        "Hi there, I'm Alex",
        "I need to set an alarm called workout",
        "7:30 AM", 
        "tomorrow",
        "yes",
        "show my alarms",
        "extend workout alarm by 10 minutes",
        "yes"
    ]
    
    demo_conversation = []
    for user_input in demo_inputs:
        response = fsm.process_input(user_input)
        demo_conversation.append({
            "user": user_input,
            "bot": response,
            "state": fsm.state,
            "entities": fsm.entities.copy()
        })
    
    return jsonify({
        "demo_conversation": demo_conversation,
        "final_stats": fsm.get_conversation_stats()
    })

@app.route('/check-alarms', methods=['GET'])
def check_alarms():
    """Check if any alarms are due and return ringing status."""
    try:
        triggered = alarm_manager.check_alarms()
        if triggered:
            return jsonify({
                'status': 'ringing',
                'message': 'Hey! Your alarm is ringing!',
                'label': triggered.get('label', 'Alarm'),
                'scheduled_for': triggered.get('scheduled_for'),
                'repeat': triggered.get('repeat', 'None')
            })
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/alarms', methods=['GET'])
def list_alarms():
    """Return all alarms from DB; if none, explicit empty array."""
    alarms = alarm_manager.get_all_alarms()
    return jsonify({ 'alarms': alarms, 'count': len(alarms) })

@app.route('/alarms', methods=['POST'])
def create_alarm():
    data = request.get_json() or {}
    label = data.get('label', 'Alarm')
    time_str = data.get('time', '')
    date_str = data.get('date', '')
    repeat = data.get('repeat', 'None')
    alarm = alarm_manager.add_alarm(time_str=time_str, date_str=date_str, repeat_str=repeat, label=label)
    return jsonify({ 'alarm': alarm }), 201

@app.route('/alarms/<int:alarm_id>', methods=['PUT'])
def update_alarm(alarm_id: int):
    data = request.get_json() or {}
    ok = alarm_manager.update_alarm(
        alarm_id,
        label=data.get('label'),
        time_str=data.get('time'),
        date_str=data.get('date'),
        repeat_str=data.get('repeat'),
        status=data.get('status')
    )
    if not ok:
        return jsonify({ 'error': 'Alarm not found or no fields to update' }), 404
    return jsonify({ 'status': 'updated' })

@app.route('/alarms/<int:alarm_id>', methods=['DELETE'])
def delete_alarm(alarm_id: int):
    ok = alarm_manager.delete_alarm(alarm_id)
    if not ok:
        return jsonify({ 'error': 'Alarm not found' }), 404
    return jsonify({ 'status': 'deleted' })

@app.route('/alarms/delete-by-criteria', methods=['POST'])
def delete_by_criteria():
    data = request.get_json() or {}
    label = data.get('label')
    time_str = data.get('time')
    date_str = data.get('date')
    deleted = alarm_manager.delete_alarms(label=label, time_str=time_str, date_str=date_str)
    if deleted == 0:
        return jsonify({ 'error': 'No matching alarm found' }), 404
    return jsonify({ 'status': 'deleted', 'count': deleted })

@app.route('/alarms/update-by-criteria', methods=['POST'])
def update_by_criteria():
    data = request.get_json() or {}
    label = data.get('label')
    new_time = data.get('new_time')
    new_date = data.get('new_date')
    new_repeat = data.get('new_repeat')
    new_status = data.get('new_status')
    updated = alarm_manager.update_alarms(label=label, new_time=new_time, new_date=new_date, new_repeat=new_repeat, new_status=new_status)
    if updated == 0:
        return jsonify({ 'error': 'No matching alarm found to update' }), 404
    return jsonify({ 'status': 'updated', 'count': updated })

if __name__ == '__main__':
    print("🚀 Enhanced Conversational Alarm Flask App Starting...")
    print("📱 Features:")
    print("   • Multi-state conversation flow")
    print("   • Natural language processing")
    print("   • Entity validation & collection")
    print("   • SQL query generation")
    print("   • Session management")
    print("   • Real-time conversation stats")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)