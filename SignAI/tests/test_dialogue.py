import unittest
import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.dialogue_manager import DialogueManager

class TestDialogueManager(unittest.TestCase):
    def setUp(self):
        self.dm = DialogueManager()

    def test_add_message(self):
        entry = self.dm.add_message("patient", "أشعر بألم خفيف")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["role"], "patient")
        self.assertEqual(entry["text"], "أشعر بألم خفيف")
        self.assertEqual(len(self.dm.get_messages()), 1)

    def test_add_empty_message_ignored(self):
        entry = self.dm.add_message("doctor", "   ")
        self.assertIsNone(entry)
        self.assertEqual(len(self.dm.get_messages()), 0)

    def test_delete_last(self):
        self.dm.add_message("patient", "أول رسالة")
        self.dm.add_message("doctor", "ثاني رسالة")
        popped = self.dm.delete_last()
        self.assertEqual(popped["text"], "ثاني رسالة")
        self.assertEqual(len(self.dm.get_messages()), 1)

    def test_clear(self):
        self.dm.add_message("patient", "رسالة")
        self.dm.clear()
        self.assertEqual(len(self.dm.get_messages()), 0)

    def test_export_transcript(self):
        self.dm.add_message("patient", "أشعر بألم في أسناني")
        self.dm.add_message("doctor", "سأقوم بفحصها حالاً")
        filepath = self.dm.export_transcript(output_dir="tests/output_transcripts")
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("المريض", content)
            self.assertIn("الطبيب", content)
            self.assertIn("أشعر بألم في أسناني", content)
        # Clean up
        try:
            os.remove(filepath)
            os.rmdir("tests/output_transcripts")
        except Exception:
            pass

if __name__ == '__main__':
    unittest.main()
