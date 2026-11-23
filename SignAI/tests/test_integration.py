import unittest
import os
import sys
from tkinter import Tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_modern import SignAIModernApp

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.root = Tk()
        self.root.withdraw()  # Headless mode: hide root window
        self.app = SignAIModernApp(self.root, start_loop=False)

    def tearDown(self):
        try:
            self.app.close()
        except Exception:
            pass

    def test_initial_state(self):
        self.assertEqual(self.app.current_mode, "WORDS")
        self.assertEqual(self.app.expression, "")
        self.assertTrue(self.app.tts_enabled_var.get())
        self.assertEqual(self.app.hold_threshold_var.get(), 1.5)

    def test_mode_toggle(self):
        self.app._toggle_mode()
        self.assertEqual(self.app.current_mode, "LETTERS")
        self.app._toggle_mode()
        self.assertEqual(self.app.current_mode, "WORDS")

    def test_composer_actions(self):
        self.app._enter_text("ألم ")
        self.assertEqual(self.app.expression, "ألم ")
        self.app._enter_text("خفيف ")
        self.assertEqual(self.app.expression, "ألم خفيف ")
        self.app._delete_char()
        self.assertEqual(self.app.expression, "ألم ")
        self.app._clear_composer()
        self.assertEqual(self.app.expression, "")

    def test_submission_to_transcript(self):
        self.app._enter_text("أشعر بألم")
        self.app._submit_sentence()
        self.assertEqual(self.app.expression, "")
        messages = self.app.dialogue_mgr.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["text"], "أشعر بألم")
        self.assertEqual(messages[0]["role"], "patient")

    def test_doctor_preset_response(self):
        self.app._add_doctor_message("افتح فمك من فضلك")
        messages = self.app.dialogue_mgr.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["text"], "افتح فمك من فضلك")
        self.assertEqual(messages[0]["role"], "doctor")

if __name__ == '__main__':
    unittest.main()
