# models/fsm.py
from models.ner_model import predict as ner_predict
from models.intent_model import predict as intent_predict
from datetime import datetime, timedelta
import re
import random
import json


class ConversationalAlarmFSM:
    def __init__(self):
        # Core FSM state
        self.state = "IDLE"
        self.current_intent = None
        self.entities = {}
        self.missing_fields = []
        self.conversation_context = {}
        self.story_progress = []

        # Conversation memory
        self.user_name = None
        self.last_alarm_created = None
        self.conversation_history = []
        self.retry_count = 0
        self.max_retries = 3

        # Intent definitions with stories
        self.intent_stories = {
            "set_alarm": {
                "required_fields": ["label", "time"],
                "optional_fields": ["date", "repeat"],
                "story_flow": ["greet_intent", "collect_entities", "confirm_action", "execute_action", "farewell"],
                "prompts": {
                    "label": [
                        "What would you like to call this alarm?",
                        "Give your alarm a name (like 'workout' or 'meeting'):",
                        "What should I label this alarm as?",
                        "How would you like me to identify this alarm?"
                    ],
                    "time": [
                        "What time should this alarm ring?",
                        "When do you want to be reminded?",
                        "At what time should I wake you up?",
                        "Please tell me the alarm time:"
                    ],
                    "date": [
                        "Which day is this alarm for?",
                        "What date should I set this for?",
                        "When do you need this reminder?",
                        "Please specify the date:"
                    ],
                    "repeat": [
                        "Should this alarm repeat? (daily, weekly, etc.)",
                        "How often should this alarm ring?",
                        "Would you like this to be a recurring alarm?",
                        "Please specify the repeat pattern:"
                    ]
                }
            },
            "cancel_alarm": {
                "required_fields": ["label"],
                "optional_fields": [],
                "story_flow": ["greet_intent", "collect_entities", "confirm_action", "execute_action", "farewell"],
                "prompts": {
                    "label": [
                        "Which alarm would you like me to cancel?",
                        "Tell me the name of the alarm to delete:",
                        "What's the label of the alarm you want removed?",
                        "Which alarm should I cancel for you?"
                    ]
                }
            },
            "update_alarm": {
                "required_fields": ["label", "time"],
                "optional_fields": ["date"],
                "story_flow": ["greet_intent", "collect_entities", "confirm_action", "execute_action", "farewell"],
                "prompts": {
                    "label": [
                        "Which alarm needs updating?",
                        "Tell me the name of the alarm to modify:",
                        "What's the label of the alarm you want to change?",
                        "Which alarm should I update?"
                    ],
                    "time": [
                        "What's the new time for this alarm?",
                        "When should I reschedule this alarm to?",
                        "Please provide the updated time:",
                        "What time should I change it to?"
                    ]
                }
            },
            "show_alarms": {
                "required_fields": [],
                "optional_fields": ["label", "date"],
                "story_flow": ["greet_intent", "execute_action", "farewell"],
                "prompts": {}
            },
            "extend_alarm": {
                "required_fields": ["label", "time"],
                "optional_fields": [],
                "story_flow": ["greet_intent", "collect_entities", "confirm_action", "execute_action", "farewell"],
                "prompts": {
                    "label": [
                        "Which alarm do you want to extend?",
                        "Tell me the alarm name to snooze:",
                        "What's the label of the alarm to delay?",
                        "Which alarm needs more time?"
                    ],
                    "time": [
                        "How much time should I add? (e.g., '10 minutes')",
                        "How long should I extend it for?",
                        "Please specify the snooze duration:",
                        "By how much should I delay this alarm?"
                    ]
                }
            },
            "repeat_alarm": {
                "required_fields": ["label", "repeat"],
                "optional_fields": [],
                "story_flow": ["greet_intent", "collect_entities", "confirm_action", "execute_action", "farewell"],
                "prompts": {
                    "label": [
                        "Which alarm should I make recurring?",
                        "Tell me the alarm to set on repeat:",
                        "What's the name of the alarm to repeat?",
                        "Which alarm needs a repeat pattern?"
                    ],
                    "repeat": [
                        "How often should this alarm repeat?",
                        "What's the repetition schedule? (daily, weekly, etc.)",
                        "Please specify the repeat frequency:",
                        "How frequently should this alarm ring?"
                    ]
                }
            },
            "start_alarm": {
                "required_fields": ["label"],
                "optional_fields": [],
                "story_flow": ["greet_intent", "collect_entities", "confirm_action", "execute_action", "farewell"],
                "prompts": {
                    "label": [
                        "Which alarm should I activate?",
                        "Tell me the alarm name to start:",
                        "What's the label of the alarm to turn on?",
                        "Which alarm should I enable?"
                    ]
                }
            },
            "stop_alarm": {
                "required_fields": ["label"],
                "optional_fields": [],
                "story_flow": ["greet_intent", "collect_entities", "confirm_action", "execute_action", "farewell"],
                "prompts": {
                    "label": [
                        "Which alarm should I stop?",
                        "Tell me the alarm name to deactivate:",
                        "What's the label of the alarm to turn off?",
                        "Which alarm should I disable?"
                    ]
                }
            }
        }

        # Conversational responses
        self.responses = {
            "greetings": [
                "Hi there! I'm here to help with your alarms. What would you like to do?",
                "Hello! Ready to manage your alarms. How can I assist you?",
                "Hey! I'm your alarm assistant. What can I help you with today?",
                "Welcome! Let's set up your perfect alarm schedule. What do you need?"
            ],
            "acknowledgments": [
                "Got it!", "Perfect!", "Understood!", "Alright!", "Excellent!",
                "Great choice!", "Sounds good!", "That works!", "Nice!"
            ],
            "confirmations": [
                "Let me confirm this for you:",
                "Just to double-check:",
                "Please verify these details:",
                "Does this look correct to you?",
                "Is this what you wanted?"
            ],
            "success": [
                "✅ Done! Your alarm is all set.",
                "🎯 Perfect! Everything's configured.",
                "⭐ Excellent! Your alarm is ready.",
                "🚀 Great! All systems go for your alarm."
            ],
            "errors": [
                "Hmm, I didn't quite catch that. Could you try again?",
                "Sorry, I'm not sure I understand. Can you rephrase?",
                "Let me help you with that. Could you be more specific?",
                "I want to make sure I get this right. Can you clarify?"
            ],
            "farewells": [
                "You're all set! Have a great day! 😊",
                "Perfect! Your alarms are ready. Anything else?",
                "All done! Sweet dreams (or productive day)! ✨",
                "Fantastic! Your alarm assistant signing off. 👋"
            ]
        }

        # Action mapping
        self.action_mapping = {
            "set_alarm": "create_alarm",
            "cancel_alarm": "delete_alarm",
            "update_alarm": "update_alarm",
            "show_alarms": "list_alarms",
            "extend_alarm": "extend_alarm",
            "repeat_alarm": "set_repeat_alarm",
            "start_alarm": "activate_alarm",
            "stop_alarm": "deactivate_alarm"
        }

    def reset(self):
        """Reset conversation while maintaining user context"""
        self.state = "IDLE"
        self.current_intent = None
        self.entities = {}
        self.missing_fields = []
        self.story_progress = []
        self.retry_count = 0

    def get_random_response(self, category):
        """Get a random response from a category"""
        return random.choice(self.responses.get(category, ["Let's continue!"]))

    def extract_user_name(self, text):
        """Try to extract user's name from input"""
        name_patterns = [
            r"i'm ([a-zA-Z]+)", r"i am ([a-zA-Z]+)",
            r"my name is ([a-zA-Z]+)", r"call me ([a-zA-Z]+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).title()
        return None

    def personalize_response(self, response):
        """Add personalization to responses"""
        if self.user_name:
            personal_starters = [f"Hi {self.user_name}!", f"{self.user_name},", f"Hey {self.user_name},"]
            if any(starter in response for starter in ["Hi there!", "Hello!", "Hey!"]):
                response = response.replace("Hi there!", f"Hi {self.user_name}!")
                response = response.replace("Hello!", f"Hello {self.user_name}!")
                response = response.replace("Hey!", f"Hey {self.user_name}!")
        return response

    def enhanced_intent_predict(self, user_input):
        """Enhanced intent prediction with conversational awareness"""
        # Handle conversational intents first
        text_lower = user_input.lower().strip()

        # Greeting detection
        if any(word in text_lower for word in ["hi", "hello", "hey", "good morning", "good evening"]):
            if not self.user_name:
                self.user_name = self.extract_user_name(text_lower)

        # Confirmation/Denial
        if self.state == "CONFIRMING":
            if any(word in text_lower for word in ["yes", "y", "confirm", "correct", "right", "ok", "sure"]):
                return "confirm_action"
            elif any(word in text_lower for word in ["no", "n", "cancel", "wrong", "incorrect"]):
                return "deny_action"

        # Try model prediction first
        try:
            predicted_intent = intent_predict(user_input)
            intent_mapping = {
                "SetAlarm": "set_alarm", "CancelAlarm": "cancel_alarm",
                "UpdateAlarm": "update_alarm", "ShowAlarms": "show_alarms",
                "ExtendAlarm": "extend_alarm", "RepeatAlarm": "repeat_alarm",
                "StartAlarm": "start_alarm", "StopAlarm": "stop_alarm"
            }
            return intent_mapping.get(predicted_intent, self.pattern_based_intent(user_input))
        except:
            return self.pattern_based_intent(user_input)

    def pattern_based_intent(self, text):
        """Pattern-based intent detection with conversational context"""
        text_lower = text.lower()

        intent_patterns = {
            "set_alarm": [r'set.*alarm', r'create.*alarm', r'new.*alarm', r'add.*alarm',
                          r'schedule.*alarm', r'wake.*me', r'remind.*me', r'alarm.*for'],
            "cancel_alarm": [r'cancel.*alarm', r'delete.*alarm', r'remove.*alarm',
                             r'turn.*off.*alarm', r'stop.*alarm', r'kill.*alarm'],
            "update_alarm": [r'update.*alarm', r'change.*alarm', r'modify.*alarm',
                             r'move.*alarm', r'reschedule.*alarm', r'edit.*alarm'],
            "show_alarms": [r'show.*alarm', r'list.*alarm', r'display.*alarm',
                            r'what.*alarm', r'my.*alarm', r'check.*alarm'],
            "extend_alarm": [r'extend.*alarm', r'snooze.*alarm', r'delay.*alarm',
                             r'postpone.*alarm', r'push.*back'],
            "repeat_alarm": [r'repeat.*alarm', r'recurring.*alarm', r'daily.*alarm',
                             r'weekly.*alarm', r'make.*repeat'],
            "start_alarm": [r'start.*alarm', r'activate.*alarm', r'turn.*on.*alarm',
                            r'enable.*alarm', r'begin.*alarm'],
            "stop_alarm": [r'stop.*alarm', r'deactivate.*alarm', r'disable.*alarm',
                           r'end.*alarm', r'silence.*alarm']
        }

        for intent, patterns in intent_patterns.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                return intent

        return "unknown"

    def enhanced_ner_predict(self, user_input):
        """Enhanced NER with conversational context"""
        try:
            raw_ner = ner_predict(user_input)
            return self.post_process_ner(raw_ner, user_input)
        except:
            return self.pattern_based_ner(user_input)

    def post_process_ner(self, ner_output, original_text):
        """Smart NER post-processing with context awareness"""
        processed = []
        original_words = original_text.lower().split()

        for token, tag in ner_output:
            token_lower = token.lower()

            # Fix repeat word context issue
            if tag == 'B-repeat' and token_lower == 'repeat':
                # Check if this is actually about setting repetition
                context_indicators = ['set', 'make', 'create', 'daily', 'weekly', 'monthly', 'every']
                if not any(indicator in original_text.lower() for indicator in context_indicators):
                    tag = 'O'

            # Enhance time detection
            elif re.match(r'\d{1,2}(:\d{2})?\s*(am|pm)', token_lower) and not tag.endswith('time'):
                tag = 'B-time'

            # Enhance date detection
            elif token_lower in ['today', 'tomorrow', 'monday', 'tuesday', 'wednesday',
                                 'thursday', 'friday', 'saturday', 'sunday'] and not tag.endswith('date'):
                tag = 'B-date'

            processed.append((token, tag))

        return processed

    def pattern_based_ner(self, text):
        """Conversational pattern-based NER"""
        tokens = text.split()
        result = []

        # Enhanced patterns with conversational awareness
        time_patterns = [
            r'\d{1,2}:\d{2}\s*(am|pm)', r'\d{1,2}\s*(am|pm)', r'\d{1,2}:\d{2}',
            r'(morning|afternoon|evening|night|noon|midnight)',
            r'\d+\s*(minutes?|hours?|mins?|hrs?|min|hr)'
        ]

        date_patterns = [
            r'(today|tomorrow|yesterday)', r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'next\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}'
        ]

        repeat_patterns = [
            r'(daily|weekly|monthly|yearly)', r'every\s+(day|week|month|year)',
            r'(weekdays|weekends)', r'once\s+a\s+(day|week|month)'
        ]

        # Conversational skip words
        skip_words = {
            'set', 'alarm', 'for', 'at', 'on', 'to', 'the', 'a', 'an', 'called', 'named',
            'please', 'can', 'could', 'would', 'will', 'i', 'want', 'need', 'like',
            'my', 'me', 'and', 'or', 'but', 'so', 'then', 'now', 'also'
        }

        for i, token in enumerate(tokens):
            token_lower = token.lower()

            # Smart entity detection with context
            if any(re.search(p, token_lower) for p in time_patterns):
                result.append((token, 'B-time'))
            elif any(re.search(p, token_lower) for p in date_patterns):
                result.append((token, 'B-date'))
            elif any(re.search(p, token_lower) for p in repeat_patterns):
                result.append((token, 'B-repeat'))
            elif (token_lower not in skip_words and len(token) > 1 and
                  re.match(r'^[a-zA-Z][a-zA-Z0-9_\-]*$', token)):
                # Context-aware label detection
                context = ' '.join(tokens[max(0, i - 2):i + 3]).lower()
                if any(indicator in context for indicator in ['called', 'named', 'for', 'alarm']):
                    result.append((token, 'B-label'))
                else:
                    result.append((token, 'O'))
            else:
                result.append((token, 'O'))

        return result

    def validate_time(self, time_str):
        """Enhanced time validation with conversational feedback"""
        if not time_str:
            return None, "I need to know what time you'd like the alarm. Could you tell me?"

        time_str = time_str.strip().lower()

        # Duration patterns (for extend_alarm)
        duration_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(sec|second|seconds)', lambda m: f"{float(m.group(1))} seconds"),
            (r'(\d+(?:\.\d+)?)\s*(min|minute|minutes)', lambda m: f"{float(m.group(1))} minutes"),
            (r'(\d+(?:\.\d+)?)\s*(hr|hrs|hour|hours)', lambda m: f"{float(m.group(1))} hours"),
        ]

        for pattern, formatter in duration_patterns:
            match = re.search(pattern, time_str)
            if match:
                return formatter(match), None

        # Natural language times
        natural_times = {
            'noon': '12:00 PM', 'midnight': '12:00 AM',
            'morning': '8:00 AM', 'afternoon': '2:00 PM',
            'evening': '6:00 PM', 'night': '10:00 PM'
        }
        if time_str in natural_times:
            return natural_times[time_str], None

        # Time format parsing
        try:
            if re.match(r'^\d{1,2}:\d{2}\s*(am|pm)$', time_str):
                return self.format_12hour_time(time_str), None
            elif re.match(r'^\d{1,2}\s*(am|pm)$', time_str):
                parts = re.match(r'^(\d{1,2})\s*(am|pm)$', time_str)
                hour, period = parts.groups()
                return f"{hour}:00 {period.upper()}", None
            elif re.match(r'^\d{1,2}:\d{2}$', time_str):
                return self.convert_24_to_12(time_str), None
            else:
                return None, "I didn't understand that time format. Try something like '7:30 AM' or '14:30'."
        except:
            return None, "That doesn't look like a valid time. Could you try again?"

    def format_12hour_time(self, time_str):
        """Format 12-hour time string"""
        parts = re.match(r'^(\d{1,2}):(\d{2})\s*(am|pm)$', time_str.lower())
        if parts:
            hour, minute, period = parts.groups()
            return f"{hour}:{minute} {period.upper()}"
        return time_str

    def convert_24_to_12(self, time_str):
        """Convert 24-hour to 12-hour format"""
        try:
            hour, minute = map(int, time_str.split(':'))
            if hour == 0:
                return f"12:{minute:02d} AM"
            elif hour < 12:
                return f"{hour}:{minute:02d} AM"
            elif hour == 12:
                return f"12:{minute:02d} PM"
            else:
                return f"{hour - 12}:{minute:02d} PM"
        except:
            return time_str

    def validate_date(self, date_str):
        """Enhanced date validation with conversational feedback"""
        if not date_str:
            return None, "Which day would you like this alarm? You can say 'today', 'tomorrow', or a specific date."

        date_str = date_str.strip().lower()
        today = datetime.today()

        # Handle relative dates
        if date_str == "today":
            return today.strftime("%Y-%m-%d"), None
        elif date_str == "tomorrow":
            return (today + timedelta(days=1)).strftime("%Y-%m-%d"), None

        # Handle weekdays
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        if date_str in weekdays:
            target_day = weekdays.index(date_str)
            days_ahead = (target_day - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = today + timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d"), None

        return None, "I didn't understand that date. Try 'today', 'tomorrow', a day of the week, or a date like '25/12/2024'."

    def validate_repeat(self, repeat_str):
        """Enhanced repeat validation with suggestions"""
        if not repeat_str:
            return None, "How often should this alarm repeat? Try 'daily', 'weekly', 'weekdays', or 'every monday'."

        repeat_str = repeat_str.strip().lower()

        valid_patterns = {
            'daily': 'daily', 'weekly': 'weekly', 'monthly': 'monthly',
            'weekdays': 'weekdays', 'weekends': 'weekends',
            'every day': 'daily', 'every week': 'weekly'
        }

        if repeat_str in valid_patterns:
            return valid_patterns[repeat_str], None

        # Handle "every [weekday]"
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in weekdays:
            if repeat_str == f"every {day}":
                return f"every_{day}", None

        return None, "I didn't understand that repeat pattern. Try 'daily', 'weekly', 'weekdays', or 'every monday'."

    def validate_label(self, label_str):
        """Enhanced label validation with suggestions"""
        if not label_str:
            return None, "What would you like to call this alarm? Give it a name like 'workout', 'meeting', or 'medicine'."

        # Clean the label
        filler_words = {'the', 'a', 'an', 'my', 'this', 'called', 'named', 'alarm'}
        words = []
        for word in label_str.strip().split():
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            if clean_word not in filler_words and len(clean_word) > 0:
                words.append(word)

        if not words:
            return None, "I need a name for this alarm. What should I call it?"

        return ' '.join(words[:3]), None

    def get_entity_prompt(self, field):
        """Get contextual prompt for missing entity"""
        if self.current_intent in self.intent_stories:
            prompts = self.intent_stories[self.current_intent]["prompts"].get(field, [])
            if prompts:
                return random.choice(prompts)
        return f"Please provide {field}:"

    def generate_confirmation(self):
        """Generate conversational confirmation message"""
        confirmation_start = random.choice(self.responses["confirmations"])

        # Build confirmation details
        details = []
        if "label" in self.entities:
            details.append(f"🏷️ Alarm: '{self.entities['label']}'")
        if "time" in self.entities:
            details.append(f"⏰ Time: {self.entities['time']}")
        if "date" in self.entities and self.entities["date"] != datetime.now().strftime("%Y-%m-%d"):
            details.append(f"📅 Date: {self.entities['date']}")
        if "repeat" in self.entities:
            details.append(f"🔁 Repeat: {self.entities['repeat']}")

        details_text = "\n".join(details)
        return f"{confirmation_start}\n\n{details_text}\n\n💬 Should I go ahead? (yes/no)"

    def process_input(self, user_input):
        """Main conversational FSM processing"""
        user_input = user_input.strip()

        # Store in conversation history
        self.conversation_history.append({"user": user_input, "timestamp": datetime.now()})

        # Handle special commands
        if user_input.lower() in ["help", "?", "what can you do"]:
            return self.show_help()
        elif user_input.lower() in ["reset", "start over", "clear"]:
            self.reset()
            return self.get_random_response("greetings")

        if self.state == "IDLE":
            return self.handle_initial_input(user_input)
        elif self.state == "COLLECTING_ENTITIES":
            return self.handle_entity_collection(user_input)
        elif self.state == "CONFIRMING":
            return self.handle_confirmation(user_input)

        return self.get_random_response("errors")

    def handle_initial_input(self, user_input):
        """Handle initial input with conversational flow"""
        # Detect intent
        self.current_intent = self.enhanced_intent_predict(user_input)

        if self.current_intent == "unknown":
            return f"{self.get_random_response('errors')} Try something like 'set workout alarm for 7 AM' or 'show my alarms'."

        # Extract entities
        ner_output = self.enhanced_ner_predict(user_input)
        entities = {}
        for token, tag in ner_output:
            tag = tag.lower()
            if tag.startswith('b-'):
                entities[tag[2:]] = token
            elif tag.startswith('i-') and tag[2:] in entities:
                entities[tag[2:]] += ' ' + token
        self.entities = entities

        # Validate entities
        for entity_type in ['time', 'date', 'repeat', 'label']:
            if entity_type in self.entities:
                val, err = getattr(self, f'validate_{entity_type}')(self.entities[entity_type])
                if err:
                    self.entities.pop(entity_type)
                else:
                    self.entities[entity_type] = val

        # Check for missing required fields
        story = self.intent_stories.get(self.current_intent, {})
        required_fields = story.get("required_fields", [])
        self.missing_fields = [f for f in required_fields if f not in self.entities]

        # Handle special case: show_alarms doesn't need confirmation
        if self.current_intent == "show_alarms":
            return self.execute_action()

        # Progress story flow
        if self.missing_fields:
            self.state = "COLLECTING_ENTITIES"
            acknowledgment = random.choice(self.responses["acknowledgments"])
            prompt = self.get_entity_prompt(self.missing_fields[0])
            return f"{acknowledgment} {prompt}"
        else:
            self.state = "CONFIRMING"
            return self.generate_confirmation()

    def handle_entity_collection(self, user_input):
        """Handle entity collection with conversational flow"""
        if not self.missing_fields:
            self.state = "CONFIRMING"
            return self.generate_confirmation()

        field = self.missing_fields[0]

        # Validate the input
        val, err = getattr(self, f'validate_{field}')(user_input)

        if err:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                self.reset()
                return "I'm having trouble understanding. Let's start fresh. What would you like to do?"
            return f"{err} Let's try again:"

        # Success!
        self.entities[field] = val
        self.missing_fields.remove(field)
        self.retry_count = 0

        acknowledgment = random.choice(self.responses["acknowledgments"])

        if self.missing_fields:
            next_prompt = self.get_entity_prompt(self.missing_fields[0])
            return f"{acknowledgment} {field.title()}: {val}\n\n{next_prompt}"
        else:
            self.state = "CONFIRMING"
            return f"{acknowledgment} {field.title()}: {val}\n\n{self.generate_confirmation()}"

    def handle_confirmation(self, user_input):
        """Handle confirmation with conversational responses"""
        intent = self.enhanced_intent_predict(user_input)

        if intent == "confirm_action" or user_input.lower().strip() in ["yes", "y", "ok", "sure", "confirm"]:
            result = self.execute_action()
            self.reset()
            return result
        elif intent == "deny_action" or user_input.lower().strip() in ["no", "n", "cancel", "nope"]:
            self.reset()
            return "No problem! What else can I help you with?"
        else:
            return "I didn't catch that. Please say 'yes' to confirm or 'no' to cancel."

    def show_help(self):
        """Show conversational help"""
        help_text = """🤖 **Your Personal Alarm Assistant**

I can help you with all your alarm needs! Here's what I can do:

**Setting Alarms** 🔔
• "Set workout alarm for 7 AM"
• "Create meeting reminder for 2:30 PM tomorrow"
• "Add daily alarm called medicine for 8 AM"

**Managing Alarms** ⚙️
• "Cancel my workout alarm"
• "Update meeting alarm to 3 PM"
• "Show all my alarms"

**Advanced Features** ✨
• "Extend workout alarm by 10 minutes"
• "Make my medicine alarm repeat daily"
• "Start my morning alarm"
• "Stop the workout alarm"

**Natural Commands** 💬
Just talk naturally! I understand:
• Time formats: "7 AM", "14:30", "noon", "evening"
• Dates: "today", "tomorrow", "monday", "next friday"
• Repetition: "daily", "weekly", "every monday", "weekdays"

Ready to help! What would you like to do? 😊"""
        return help_text

    def execute_action(self):
        """Execute action with conversational response"""
        sql_query = self.generate_sql()
        action_type = self.action_mapping.get(self.current_intent, "unknown_action")

        # Generate conversational success response
        response = self.generate_success_response()

        # Store last action for context
        self.last_alarm_created = {
            "intent": self.current_intent,
            "entities": self.entities.copy(),
            "timestamp": datetime.now()
        }

        return f"{response}\n\n💾 SQL: {sql_query}\n\n{random.choice(self.responses['farewells'])}"

    def generate_success_response(self):
        """Generate contextual success responses"""
        label = self.entities.get('label', 'alarm')
        time_val = self.entities.get('time', '')
        date_val = self.entities.get('date', '')
        repeat_val = self.entities.get('repeat', '')

        success_templates = {
            "set_alarm": [
                f"🎯 Perfect! I've created your '{label}' alarm for {time_val}" +
                (f" on {self.format_date_friendly(date_val)}" if date_val else "") +
                (f", repeating {repeat_val}" if repeat_val else "") + ".",
                f"✅ Your '{label}' alarm is all set for {time_val}" +
                (f" {self.format_date_friendly(date_val)}" if date_val else "") + "!",
                f"🚀 Done! '{label}' will wake you up at {time_val}" +
                (f" on {self.format_date_friendly(date_val)}" if date_val else "") + "."
            ],
            "cancel_alarm": [
                f"✅ I've cancelled your '{label}' alarm. No more interruptions!",
                f"🗑️ Done! The '{label}' alarm has been removed from your schedule.",
                f"👍 Perfect! Your '{label}' alarm is history."
            ],
            "update_alarm": [
                f"🔧 Great! I've updated your '{label}' alarm to {time_val}" +
                (f" on {self.format_date_friendly(date_val)}" if date_val else "") + ".",
                f"⚡ Perfect! '{label}' alarm rescheduled to {time_val}" +
                (f" {self.format_date_friendly(date_val)}" if date_val else "") + ".",
                f"✨ All set! Your '{label}' alarm is now at {time_val}."
            ],
            "show_alarms": [
                "📋 Here's your complete alarm schedule:",
                "⏰ Your current alarms:",
                "📱 Here are all your active alarms:"
            ],
            "extend_alarm": [
                f"😴 No problem! I've given you {time_val} more sleep time for '{label}'.",
                f"⏰ Extended! Your '{label}' alarm will ring {time_val} later.",
                f"🛌 Sweet dreams! '{label}' alarm snoozed for {time_val}."
            ],
            "repeat_alarm": [
                f"🔁 Excellent! Your '{label}' alarm will now repeat {repeat_val}.",
                f"📅 Perfect! I've set '{label}' to ring {repeat_val}.",
                f"✨ Great choice! '{label}' is now a {repeat_val} alarm."
            ],
            "start_alarm": [
                f"▶️ Your '{label}' alarm is now active and ready!",
                f"🔥 Activated! '{label}' alarm is running.",
                f"🚀 Perfect! '{label}' alarm is now live."
            ],
            "stop_alarm": [
                f"⏹️ Silenced! Your '{label}' alarm is now off.",
                f"🔇 Done! '{label}' alarm has been stopped.",
                f"💤 All quiet! '{label}' alarm is deactivated."
            ]
        }

        templates = success_templates.get(self.current_intent, ["Action completed successfully!"])
        return random.choice(templates)

    def format_date_friendly(self, date_str):
        """Convert date to friendly format"""
        if not date_str:
            return ""

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.now()

            if date_obj.date() == today.date():
                return "today"
            elif date_obj.date() == (today + timedelta(days=1)).date():
                return "tomorrow"
            else:
                return date_obj.strftime("%A, %B %d")
        except:
            return date_str

    def generate_sql(self):
        """Generate SQL with enhanced formatting"""
        if not self.entities:
            return ""

        def sql_escape(value):
            return value.replace("'", "''") if isinstance(value, str) else str(value)

        clean_entities = {}
        for k, v in self.entities.items():
            if v and str(v).strip():
                clean_entities[k] = sql_escape(str(v).strip())

        # SQL generation based on intent
        if self.current_intent == "set_alarm":
            cols = []
            vals = []
            for k, v in clean_entities.items():
                cols.append(k)
                vals.append(f"'{v}'")

            # Add default date if not provided
            if 'date' not in clean_entities:
                cols.append('date')
                vals.append(f"'{datetime.now().strftime('%Y-%m-%d')}'")

            return f"INSERT INTO alarms ({', '.join(cols)}) VALUES ({', '.join(vals)});"

        elif self.current_intent == "cancel_alarm":
            label = clean_entities.get('label', 'alarm')
            conditions = [f"label = '{label}'"]

            for field in ['time', 'date']:
                if field in clean_entities:
                    conditions.append(f"{field} = '{clean_entities[field]}'")

            return f"DELETE FROM alarms WHERE {' AND '.join(conditions)};"

        elif self.current_intent == "update_alarm":
            label = clean_entities.get('label', 'alarm')
            updates = []

            for field in ['time', 'date', 'repeat']:
                if field in clean_entities:
                    updates.append(f"{field} = '{clean_entities[field]}'")

            if not updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")

            return f"UPDATE alarms SET {', '.join(updates)} WHERE label = '{label}';"

        elif self.current_intent == "show_alarms":
            base_query = "SELECT label, time, date, repeat, status FROM alarms"
            conditions = []

            for field in ['label', 'date', 'status']:
                if field in clean_entities:
                    conditions.append(f"{field} = '{clean_entities[field]}'")

            if conditions:
                base_query += f" WHERE {' AND '.join(conditions)}"

            return base_query + " ORDER BY date ASC, time ASC;"

        elif self.current_intent == "extend_alarm":
            label = clean_entities.get('label', 'alarm')
            extension = clean_entities.get('time', '10 minutes')

            return f"""UPDATE alarms 
                      SET extended_by = '{extension}', 
                          status = 'extended', 
                          extended_at = CURRENT_TIMESTAMP 
                      WHERE label = '{label}' AND status = 'active';"""

        elif self.current_intent == "repeat_alarm":
            label = clean_entities.get('label', 'alarm')
            repeat_pattern = clean_entities.get('repeat', 'daily')

            return f"""UPDATE alarms 
                      SET repeat_pattern = '{repeat_pattern}', 
                          updated_at = CURRENT_TIMESTAMP 
                      WHERE label = '{label}';"""

        elif self.current_intent == "start_alarm":
            label = clean_entities.get('label', 'alarm')

            return f"""UPDATE alarms 
                      SET status = 'active', 
                          started_at = CURRENT_TIMESTAMP 
                      WHERE label = '{label}';"""

        elif self.current_intent == "stop_alarm":
            label = clean_entities.get('label', 'alarm')

            return f"""UPDATE alarms 
                      SET status = 'stopped', 
                          stopped_at = CURRENT_TIMESTAMP 
                      WHERE label = '{label}' AND status IN ('active', 'ringing');"""

        return f"-- No SQL generated for intent: {self.current_intent}"

    def get_conversation_stats(self):
        """Get conversation statistics for debugging/monitoring"""
        return {
            "current_state": self.state,
            "current_intent": self.current_intent,
            "entities_collected": len(self.entities),
            "missing_fields": len(self.missing_fields),
            "conversation_turns": len(self.conversation_history),
            "user_name": self.user_name,
            "retry_count": self.retry_count
        }

    def export_conversation_log(self):
        """Export conversation for analysis"""
        return {
            "session_id": id(self),
            "user_name": self.user_name,
            "conversation_history": self.conversation_history,
            "last_action": self.last_alarm_created,
            "stats": self.get_conversation_stats(),
            "timestamp": datetime.now().isoformat()
        }


# Example usage and test function
def demo_conversation():
    """Demo conversation flow"""
    fsm = ConversationalAlarmFSM()

    test_inputs = [
        "Hi there, I'm Sarah",
        "I need to set an alarm",
        "workout",
        "7 AM",
        "tomorrow",
        "yes",
        "show my alarms",
        "cancel the workout alarm",
        "yes"
    ]

    print("🎭 **RASA-Style Conversation Demo**")
    print("=" * 50)

    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n👤 User: {user_input}")
        response = fsm.process_input(user_input)
        print(f"🤖 Bot: {response}")

        if i == 3:  # After collecting some info
            print(f"📊 Stats: {fsm.get_conversation_stats()}")

    print(f"\n📋 **Final Conversation Log:**")
    print(json.dumps(fsm.export_conversation_log(), indent=2, default=str))


if __name__ == "__main__":
    demo_conversation()