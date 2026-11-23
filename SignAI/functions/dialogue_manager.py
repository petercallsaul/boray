import os
import time
from datetime import datetime

class DialogueManager:
    """
    Manages structured, bidirectional clinical dialogue between Doctor and Patient.
    Supports real-time logging, role categorization, and clinical transcript export.
    """
    CLINICAL_DOCTOR_PRESETS = [
        "أهلاً بك، تفضل بالجلوس",
        "هل تشعر بألم في هذه المنطقة؟",
        "افتح فمك من فضلك",
        "سنأخذ استراحة قصيرة",
        "سأقوم بوضع مخدر موضعي خفيف",
        "تم الانتهاء بنجاح، يمكنك غسل فمك الآن",
    ]

    def __init__(self):
        self.messages = []
        self._msg_counter = 0

    def add_message(self, role, text):
        """
        Adds a message to the dialogue transcript.
        role: 'patient', 'doctor', or 'system'
        text: Arabic string
        """
        cleaned = text.strip()
        if not cleaned:
            return None

        self._msg_counter += 1
        entry = {
            "id": self._msg_counter,
            "role": role,
            "text": cleaned,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
        self.messages.append(entry)
        return entry

    def clear(self):
        """Clears all conversation messages."""
        self.messages = []
        self._msg_counter = 0

    def delete_last(self):
        """Removes the most recent message."""
        if self.messages:
            return self.messages.pop()
        return None

    def get_messages(self):
        """Returns the full message list."""
        return self.messages

    def export_transcript(self, output_dir="transcripts"):
        """
        Exports the current dialogue ledger into a formatted Markdown medical record.
        Returns the absolute filepath of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_transcript_{timestamp_str}.md"
        filepath = os.path.join(output_dir, filename)

        patient_count = sum(1 for m in self.messages if m["role"] == "patient")
        doctor_count = sum(1 for m in self.messages if m["role"] == "doctor")

        lines = [
            "# تقرير محادثة عيادة الأسنان - SignAI Clinical Record",
            f"**تاريخ الجلسة**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**إجمالي الرسائل**: {len(self.messages)} (المريض: {patient_count} | الطبيب: {doctor_count})",
            "",
            "---",
            "",
            "## سجل الحوار المتبادل",
            "",
        ]

        if not self.messages:
            lines.append("*لا توجد رسائل مسجلة في هذه الجلسة.*")
        else:
            for msg in self.messages:
                speaker_label = "المريض" if msg["role"] == "patient" else ("الطبيب" if msg["role"] == "doctor" else "النظام")
                lines.append(f"**[{msg['timestamp']}] {speaker_label}**: {msg['text']}")
                lines.append("")

        lines.extend([
            "---",
            "",
            "## ملاحظات وإرشادات",
            "- تم إنشاء هذا التقرير تلقائياً بواسطة نظام SignAI للترجمة الفورية للغة الإشارة.",
            "- يحتفظ التقرير بتسلسل استجابات المريض الطبية لتسهيل المتابعة السريرية.",
            "",
            f"*SignAI System Version 2.0 | High-Craft Clinical Edition*",
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return os.path.abspath(filepath)
