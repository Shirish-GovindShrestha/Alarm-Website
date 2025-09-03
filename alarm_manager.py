from datetime import datetime, timedelta
import sqlite3
import re
from typing import List, Dict, Optional, Tuple
import os


class AlarmManager:
    """In-memory alarm manager for simple scheduling and checks."""

    def __init__(self) -> None:
        # SQLite database path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._db_path = os.path.join(base_dir, 'alarms.db')
        self._ensure_schema()

    def add_alarm(self, time_str: str, date_str: str, repeat_str: str = 'None', label: str = 'Alarm') -> Dict:
        """Add an alarm using human-friendly time/date strings.

        - time_str examples: '07:00', '7:30', '7:30 AM', '07:00 PM'
        - date_str examples: 'today', 'tomorrow', '2025-08-19'
        - repeat_str examples: 'None', 'daily'
        """
        trigger_dt = self._parse_datetime(time_str, date_str)
        now_minute = datetime.now().replace(second=0, microsecond=0)
        # If date not specified, 'today', or equals today's ISO, and the time already passed, schedule for tomorrow
        normalized_date = (date_str or '').strip().lower()
        today_iso = datetime.now().strftime('%Y-%m-%d').lower()
        if trigger_dt.replace(second=0, microsecond=0) <= now_minute and (normalized_date in {'', 'today'} or normalized_date == today_iso):
            trigger_dt = (trigger_dt + timedelta(days=1)).replace(second=0, microsecond=0)
        created_at = datetime.now()
        repeat_norm = (repeat_str or 'None').lower()
        with sqlite3.connect(self._db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO alarms (label, time, date, repeat, status, next_trigger, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label or 'Alarm',
                    time_str,
                    date_str,
                    repeat_norm,
                    'active',
                    trigger_dt.replace(second=0, microsecond=0).isoformat(),
                    created_at.isoformat(),
                )
            )
            alarm_id = c.lastrowid
        return {
            'id': alarm_id,
            'label': label or 'Alarm',
            'time': time_str,
            'date': date_str,
            'repeat': repeat_norm,
            'status': 'active',
            'next_trigger': trigger_dt.isoformat(),
            'created_at': created_at.strftime('%H:%M:%S'),
        }

    def check_alarms(self) -> Optional[Dict]:
        """Check alarms and return the first triggered alarm info if any.

        For 'daily' repeats, advance the next_trigger by one day after triggering.
        """
        now_minute = datetime.now().replace(second=0, microsecond=0)
        with sqlite3.connect(self._db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT id, label, time, date, repeat, status, next_trigger
                FROM alarms
                WHERE status = 'active' AND datetime(next_trigger) <= datetime(?)
                ORDER BY datetime(next_trigger) ASC
                LIMIT 1
                """,
                (now_minute.isoformat(),)
            )
            row = c.fetchone()
            if not row:
                return None
            alarm_id, label, time_str, date_str, repeat_str, status, next_trigger_iso = row
            next_trigger_dt = datetime.fromisoformat(next_trigger_iso)

            if (repeat_str or 'None').lower() == 'daily':
                new_next = (next_trigger_dt + timedelta(days=1)).replace(second=0, microsecond=0)
                c.execute("UPDATE alarms SET next_trigger = ? WHERE id = ?", (new_next.isoformat(), alarm_id))
            else:
                c.execute("UPDATE alarms SET status = 'triggered' WHERE id = ?", (alarm_id,))

            return {
                'id': alarm_id,
                'label': label or 'Alarm',
                'scheduled_for': next_trigger_dt.isoformat(),
                'repeat': repeat_str or 'None'
            }

    def get_all_alarms(self) -> List[Dict]:
        with sqlite3.connect(self._db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT id, label, time, date, repeat, status, next_trigger, created_at
                FROM alarms
                ORDER BY datetime(created_at) DESC
                """
            )
            rows = c.fetchall()
        alarms: List[Dict] = []
        for r in rows:
            alarms.append({
                'id': r[0], 'label': r[1], 'time': r[2], 'date': r[3], 'repeat': r[4] or 'None',
                'status': r[5], 'next_trigger': r[6], 'created_at': self._fmt_time(r[7])
            })
        return alarms

    def update_alarm(self, alarm_id: int, *, label: Optional[str] = None, time_str: Optional[str] = None,
                     date_str: Optional[str] = None, repeat_str: Optional[str] = None, status: Optional[str] = None) -> bool:
        """Update an alarm by id; returns True if updated, False if missing."""
        with sqlite3.connect(self._db_path) as conn:
            c = conn.cursor()
            # Build fields
            fields: List[Tuple[str, object]] = []
            if label is not None:
                fields.append(("label", label))
            if time_str is not None:
                fields.append(("time", time_str))
            if date_str is not None:
                fields.append(("date", date_str))
            if repeat_str is not None:
                fields.append(("repeat", (repeat_str or 'None').lower()))
            if status is not None:
                fields.append(("status", status))

            # Recompute next_trigger if time/date changed
            if any(k in dict(fields) for k in ("time", "date")):
                new_time = time_str
                new_date = date_str
                # fetch current if missing
                c.execute("SELECT time, date FROM alarms WHERE id = ?", (alarm_id,))
                row = c.fetchone()
                if not row:
                    return False
                curr_time, curr_date = row
                new_trigger = self._parse_datetime(new_time or curr_time, new_date or curr_date)
                fields.append(("next_trigger", new_trigger.replace(second=0, microsecond=0).isoformat()))

            if not fields:
                return False
            set_clause = ", ".join([f"{k} = ?" for k, _ in fields])
            values = [v for _, v in fields]
            values.append(alarm_id)
            c.execute(f"UPDATE alarms SET {set_clause} WHERE id = ?", values)
            return c.rowcount > 0

    def delete_alarm(self, alarm_id: int) -> bool:
        with sqlite3.connect(self._db_path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM alarms WHERE id = ?", (alarm_id,))
            return c.rowcount > 0

    # -------- Criteria-based helpers (label/time/date) --------
    def find_alarms(self, label: Optional[str] = None, time_str: Optional[str] = None, date_str: Optional[str] = None) -> List[Dict]:
        where, params = self._build_where(label, time_str, date_str)
        with sqlite3.connect(self._db_path) as conn:
            c = conn.cursor()
            c.execute(
                f"SELECT id, label, time, date, repeat, status, next_trigger, created_at FROM alarms {where} ORDER BY datetime(created_at) DESC",
                params
            )
            rows = c.fetchall()
        return [{
            'id': r[0], 'label': r[1], 'time': r[2], 'date': r[3], 'repeat': r[4] or 'None',
            'status': r[5], 'next_trigger': r[6], 'created_at': self._fmt_time(r[7])
        } for r in rows]

    def delete_alarms(self, label: Optional[str] = None, time_str: Optional[str] = None, date_str: Optional[str] = None) -> int:
        where, params = self._build_where(label, time_str, date_str)
        with sqlite3.connect(self._db_path) as conn:
            c = conn.cursor()
            c.execute(f"DELETE FROM alarms {where}", params)
            return c.rowcount

    def update_alarms(self, label: str, *, new_time: Optional[str] = None, new_date: Optional[str] = None,
                      new_repeat: Optional[str] = None, new_status: Optional[str] = None) -> int:
        if not label:
            return 0
        updates: List[str] = []
        params: List[object] = []
        if new_time is not None:
            updates.append("time = ?")
            params.append(new_time)
        if new_date is not None:
            updates.append("date = ?")
            params.append(new_date)
        if new_repeat is not None:
            updates.append("repeat = ?")
            params.append((new_repeat or 'None').lower())
        if new_status is not None:
            updates.append("status = ?")
            params.append(new_status)

        # Recompute next_trigger if time/date provided
        recompute_trigger = (new_time is not None) or (new_date is not None)

        with sqlite3.connect(self._db_path) as conn:
            c = conn.cursor()
            if recompute_trigger:
                c.execute("SELECT id, time, date FROM alarms WHERE label = ?", (label,))
                rows = c.fetchall()
                count = 0
                for _id, curr_time, curr_date in rows:
                    t = new_time if new_time is not None else curr_time
                    d = new_date if new_date is not None else curr_date
                    next_dt = self._parse_datetime(t, d).replace(second=0, microsecond=0)
                    # If updating to a time that already passed today and date is today/empty, roll to tomorrow
                    now_minute = datetime.now().replace(second=0, microsecond=0)
                    normalized_date = (d or '').strip().lower()
                    today_iso = datetime.now().strftime('%Y-%m-%d').lower()
                    if next_dt <= now_minute and (normalized_date in {'', 'today'} or normalized_date == today_iso):
                        next_dt = (next_dt + timedelta(days=1)).replace(second=0, microsecond=0)
                    next_tr = next_dt.isoformat()
                    # When time/date change, ensure alarm is active and rescheduled
                    set_clause = ", ".join(updates + ["next_trigger = ?", "status = 'active'"]) if updates else "next_trigger = ?, status = 'active'"
                    set_params = params + [next_tr, _id]
                    c.execute(f"UPDATE alarms SET {set_clause} WHERE id = ?", set_params)
                    count += c.rowcount
                return count
            else:
                if not updates:
                    return 0
                set_clause = ", ".join(updates)
                params.append(label)
                c.execute(f"UPDATE alarms SET {set_clause} WHERE label = ?", params)
                return c.rowcount

    def _build_where(self, label: Optional[str], time_str: Optional[str], date_str: Optional[str]) -> Tuple[str, List[object]]:
        clauses: List[str] = []
        params: List[object] = []
        if label:
            clauses.append("label = ?")
            params.append(label)
        # If time/date are provided, apply exact match; otherwise match by label only
        if time_str:
            clauses.append("time = ?")
            params.append(time_str)
        if date_str:
            clauses.append("date = ?")
            params.append(date_str)
        if not clauses:
            return ("", [])
        return ("WHERE " + " AND ".join(clauses), params)

    # ------------------------
    # Helpers
    # ------------------------

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS alarms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    time TEXT,
                    date TEXT,
                    repeat TEXT,
                    status TEXT,
                    next_trigger TEXT,
                    created_at TEXT
                )
                """
            )

    def _parse_datetime(self, time_str: str, date_str: str) -> datetime:
        date = self._parse_date(date_str)
        t = self._parse_time(time_str)
        return datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=t['hour'],
            minute=t['minute'],
        )

    def _parse_date(self, date_str: str) -> datetime:
        ds = (date_str or '').strip().lower()
        now = datetime.now()
        if ds in ('today', ''):
            return now
        if ds == 'tomorrow':
            return now + timedelta(days=1)
        # Accept ISO yyyy-mm-dd
        try:
            return datetime.strptime(ds, '%Y-%m-%d')
        except Exception:
            pass
        # Fallback: try common mm/dd/yyyy or dd/mm/yyyy by heuristic (if ambiguous, treat as mm/dd)
        for fmt in ('%m/%d/%Y', '%d/%m/%Y'):
            try:
                return datetime.strptime(ds, fmt)
            except Exception:
                continue
        # Default to today if unparsable
        return now

    def _parse_time(self, time_str: str) -> Dict[str, int]:
        ts = (time_str or '').strip().lower()

        # Patterns: HH:MM, H:MM, HH:MM am/pm, H am/pm
        ampm = None
        m = re.match(r'^(\d{1,2}):(\d{2})\s*(am|pm)?$', ts)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2))
            ampm = m.group(3)
        else:
            m2 = re.match(r'^(\d{1,2})\s*(am|pm)$', ts)
            if m2:
                hour = int(m2.group(1))
                minute = 0
                ampm = m2.group(2)
            else:
                # Fallback to HH:MM only
                try:
                    parts = ts.split(':')
                    hour = int(parts[0])
                    minute = int(parts[1]) if len(parts) > 1 else 0
                except Exception:
                    # Default to next minute
                    now = datetime.now() + timedelta(minutes=1)
                    return {'hour': now.hour, 'minute': now.minute}

        if ampm:
            if ampm == 'pm' and 1 <= hour <= 11:
                hour += 12
            if ampm == 'am' and hour == 12:
                hour = 0

        hour = max(0, min(23, hour))
        minute = max(0, min(59, minute))
        return {'hour': hour, 'minute': minute}

    def _fmt_time(self, iso_or_time: Optional[str]) -> str:
        if not iso_or_time:
            return ''
        try:
            return datetime.fromisoformat(iso_or_time).strftime('%H:%M:%S')
        except Exception:
            return str(iso_or_time)


