import threading
import queue
import time

class TTSEngine:
    """
    Asynchronous, non-blocking Arabic Text-to-Speech engine.
    Runs speech synthesis on a dedicated background daemon thread to ensure 0 FPS drops in video capture.
    """
    def __init__(self, rate=145, volume=1.0, enabled=True):
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self.queue = queue.Queue()
        self._engine = None
        self._running = True
        self._arabic_voice_id = None

        # Start background worker thread
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def _init_engine(self):
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)

            # Look for Arabic voice
            voices = engine.getProperty('voices')
            for v in voices:
                v_name = v.name.lower()
                v_lang = str(getattr(v, 'languages', '')).lower()
                if 'arabic' in v_name or 'ar_sa' in v_name or 'naayf' in v_name or 'hoda' in v_name or 'ar' in v_lang:
                    self._arabic_voice_id = v.id
                    engine.setProperty('voice', v.id)
                    break
            return engine
        except Exception as e:
            print(f"[TTSEngine] Note: TTS initialization fallback ({e})")
            return None

    def _process_queue(self):
        self._engine = self._init_engine()
        while self._running:
            try:
                text = self.queue.get(timeout=0.2)
                if text is None:
                    break
                if self.enabled and self._engine is not None and text.strip():
                    try:
                        self._engine.say(text)
                        self._engine.runAndWait()
                    except Exception as e:
                        # Re-init on failure
                        print(f"[TTSEngine] Playback exception: {e}")
                        self._engine = self._init_engine()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def speak(self, text):
        """Queues an utterance for speech synthesis without blocking the caller."""
        if self.enabled and text and text.strip():
            self.queue.put(text.strip())

    def toggle(self):
        """Toggles audio speech on or off."""
        self.enabled = not self.enabled
        return self.enabled

    def stop(self):
        """Stops the speech worker thread cleanly."""
        self._running = False
        self.queue.put(None)
