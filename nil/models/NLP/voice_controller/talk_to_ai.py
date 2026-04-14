"""
Voice conversation system with zero-knowledge AI.
You talk (voice) -> AI learns -> AI responds (voice).
"""

import os
import tempfile

import speech_recognition as sr
from gtts import gTTS
from nil.models.zero_knowledge_ai import BabyAI
from playsound import playsound


class VoiceConversationAI:
    """
    Complete voice conversation system.
    You speak -> AI learns and responds.
    """

    def __init__(self, knowledge_path=None):
        # Initialize baby AI (blank slate)
        self.ai = BabyAI()

        # Load previous knowledge if exists
        if knowledge_path and os.path.exists(knowledge_path):
            self.ai.load_knowledge(knowledge_path)

        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        print("\n🎤 Voice Conversation AI Ready!")
        print("Speak to teach the AI. It starts knowing NOTHING.\n")

    def listen(self):
        """
        Listen to your voice and convert to text.
        This is the AI's ONLY source of data!
        """
        with self.microphone as source:
            print("🎤 Listening... (speak now)")

            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            try:
                # Listen
                audio = self.recognizer.listen(source, timeout=5)

                # Convert speech to text
                print("🔄 Processing...")
                text = self.recognizer.recognize_google(audio)

                return text

            except sr.WaitTimeoutError:
                print("⏱️  No speech detected")
                return None
            except sr.UnknownValueError:
                print("❌ Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"❌ Recognition service error: {e}")
                return None

    def speak(self, text):
        """AI speaks back using text-to-speech."""
        if not text or text == "...":
            return

        print(f"🤖 AI says: {text}")

        try:
            # Generate speech
            tts = gTTS(text=text, lang="en", slow=False)

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                temp_path = f.name
                tts.save(temp_path)

            # Play
            playsound(temp_path)

            # Clean up
            os.unlink(temp_path)

        except Exception as e:
            print(f"❌ TTS error: {e}")

    def converse(self, save_path="knowledge/my_ai_knowledge.json"):
        """
        Main conversation loop.
        You talk -> AI learns -> AI responds.
        """
        print("\n" + "=" * 60)
        print("VOICE CONVERSATION MODE")
        print("=" * 60)
        print("Commands:")
        print("  - Just speak normally to teach the AI")
        print("  - Say 'report' to see what AI learned")
        print("  - Say 'quit' to exit and save")
        print("=" * 60 + "\n")

        conversation_count = 0

        while True:
            # Listen to your voice
            text = self.listen()

            if text is None:
                continue

            # Check for commands
            if "quit" in text.lower() or "exit" in text.lower():
                print("\n👋 Saving and exiting...")
                self.ai.save_knowledge(save_path)
                break

            if "report" in text.lower():
                self.ai.report_learning()
                continue

            # AI learns from what you said
            self.ai.hear(text)
            conversation_count += 1

            # AI generates response (after learning a bit)
            if conversation_count >= 3:  # Give it time to learn first
                response = self.ai.think()
                self.speak(response)
            else:
                print("  🍼 Learning... (need more input to respond)")

            # Auto-save periodically
            if conversation_count % 10 == 0:
                self.ai.save_knowledge(save_path)
                print("  💾 Knowledge auto-saved")


def main():
    """Start voice conversation with blank-slate AI."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Talk to AI that learns from your voice"
    )
    parser.add_argument(
        "--knowledge",
        default="knowledge/my_ai_knowledge.json",
        help="Path to save/load AI knowledge",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start with blank AI (ignore saved knowledge)",
    )

    args = parser.parse_args()

    knowledge_path = None if args.fresh else args.knowledge

    # Create voice AI
    voice_ai = VoiceConversationAI(knowledge_path)

    # Start conversation
    voice_ai.converse(save_path=args.knowledge)


if __name__ == "__main__":
    main()
