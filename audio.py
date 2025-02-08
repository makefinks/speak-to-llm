import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Literal, Optional, Union

import numpy as np
import pyaudio
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from pynput.keyboard import Key, KeyCode, Listener
from rich.console import Console

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs


class AudioManager:
    """
    A class to manage audio recording, text-to-speech conversion, and audio playback.
    
    Depending on the TTS service specified, this class either streams audio
    via OpenAI or ElevenLabs. It also listens for keyboard input to control
    recording and playback.
    """

    def __init__(
        self,
        speech_queue: queue.Queue[str],
        console: Console,
        tts_service: Literal["openai", "elevenlabs"],
        language: Literal["en", "multi"],
        voice_id: str,
    ) -> None:
        """
        Initialize the AudioManager.

        Args:
            speech_queue: A queue containing text sentences to be spoken.
            console: A rich Console object for printing status messages.
            tts_service: The TTS service to use ("openai" or "elevenlabs").
            language: The language setting ("en" or "multi").
            voice_id: The voice identifier (used by the TTS service).
        """
        self.tts_service: Literal["openai", "elevenlabs"] = tts_service
        self.audio_stream_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self.speech_queue: queue.Queue[str] = speech_queue
        self.console: Console = console
        self.language: Literal["en", "multi"] = language
        self.voice_id: str = voice_id

        self.fs: int = 44100  # Sampling frequency for recording
        self.frames: list[np.ndarray] = []
        self.recording: bool = False
        self.playback: bool = True

        self.p: pyaudio.PyAudio = pyaudio.PyAudio()
        self.player_stream: pyaudio.Stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=8048,
        )

        self.mpv_process: Optional[subprocess.Popen] = None
        if self.tts_service == "elevenlabs":
            mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
            self.mpv_process = subprocess.Popen(
                mpv_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def on_press(self, key: Key | KeyCode) -> Optional[bool]:
        """
        Handle key press events.

        - Space: Start recording.
        - Enter: Stop recording.
        - 'q': Exit the program.
        - 's': Stop current playback and clear queues.
        - 'c': Clear the console.

        Args:
            key: The key that was pressed.

        Returns:
            False if the listener should stop, otherwise None.
        """
        if key == Key.space:
            self.recording = True
        elif key == Key.enter:
            if self.recording:
                self.recording = False
                self.playback = True
                return False  # Stop the listener
        else:
            try:
                if hasattr(key, "char"):
                    if key.char == "q":
                        os._exit(0)
                    elif key.char == "s":
                        self.console.print("[bold red]Stopping playback...")
                        self.playback = False
                        # Clear both queues
                        with self.speech_queue.mutex:
                            self.speech_queue.queue.clear()
                        with self.audio_stream_queue.mutex:
                            self.audio_stream_queue.queue.clear()

                        if self.tts_service == "openai":
                            # Fully close and reinitialize the playback stream
                            self.player_stream.stop_stream()
                            self.player_stream.close()
                            self.player_stream = self.p.open(
                                format=pyaudio.paInt16,
                                channels=1,
                                rate=24000,
                                output=True,
                                frames_per_buffer=8048,
                            )
                        else:
                            if self.mpv_process:
                                self.mpv_process.terminate()
                                self.mpv_process.wait()
                                mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
                                self.mpv_process = subprocess.Popen(
                                    mpv_command,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                )

                    elif key.char == "c":
                        self.console.clear()
            except AttributeError:
                pass

        return None

    def record_audio(self) -> Optional[str]:
        """
        Record audio from the default input device until the user stops recording.

        The recording is started when the user presses the space bar and stopped when
        the user presses enter. A temporary WAV file is created with the recorded audio.

        Returns:
            The path to the temporary WAV file containing the recording, or None if no audio was recorded.
        """

        listener = Listener(on_press=self.on_press)
        listener.start()
        try:
            self.console.print(
                "[bold blue]Press 'space' to start and 'enter' to stop the recording. "
                "Press 's' to stop TTS and 'q' to exit the program."
            )
            # Wait for the recording to start.
            while not self.recording:
                time.sleep(0.1)
                if not listener.is_alive():
                    break

            if self.recording:
                with self.console.status("[bold green]Recording...") as status:
                    with sd.InputStream(
                        samplerate=self.fs, channels=1, callback=self.callback
                    ):
                        while self.recording:
                            time.sleep(0.1)

        finally:
            listener.stop()
            listener.join()

        if not self.frames:
            self.console.print("[bold red]No audio recorded.")
            return None

        # Save the recording
        recording: np.ndarray = np.vstack(self.frames)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, recording, self.fs)
            temp_file_path = tmp_file.name

        self.frames.clear()
        return temp_file_path

    def callback(
        self, indata: np.ndarray, frame_count: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """
        Audio callback function to handle incoming audio data.

        Args:
            indata: The recorded audio data.
            frame_count: The number of frames.
            time_info: A dictionary containing timing information.
            status: A CallbackFlags object with error/overflow information.
        """
        if self.recording:
            self.frames.append(indata.copy())

    def text_to_speech(
        self, text: str, client: OpenAI | ElevenLabs
    ) -> None:
        """
        Convert the provided text to speech using the selected TTS service.

        Args:
            text: The text to convert to speech.
            client: An instance of the TTS service client.
        """
        if self.tts_service == "openai":
            with client.audio.speech.with_streaming_response.create(
                model="tts-1-hd",
                voice="nova",
                speed=1,
                response_format="pcm",
                input=text,
            ) as response:
                for chunk in response.iter_bytes(chunk_size=128):
                    self.audio_stream_queue.put(chunk)
                    
            # Append silence (converted to bytes) to ensure smooth playback.
            silence = np.zeros(4096, dtype=np.int16)
            self.audio_stream_queue.put(silence.tobytes())

        elif self.tts_service == "elevenlabs":
            audio_stream = client.generate(
                optimize_streaming_latency="2",
                model="eleven_turbo_v2" if self.language == "en" else "eleven_multilingual_v2",
                voice=self.voice_id,
                voice_settings=VoiceSettings(
                    stability=0.6,
                    similarity_boost=0.8,
                    style=0.3,
                    use_speaker_boost=True,
                ),
                text=text,
                stream=True,
            )
            for chunk in audio_stream:
                self.audio_stream_queue.put(chunk)
            # Optionally, append silence if needed.
            # silence = np.zeros(4096, dtype=np.int16)
            # self.audio_stream_queue.put(silence.tobytes())

    def audio_stream_worker(self) -> None:
        """
        Worker method to continuously write audio chunks from the audio stream queue
        to the output device.
        """
        while True:
            audio_chunk: Optional[bytes] = self.audio_stream_queue.get()
            if audio_chunk is None:
                break

            if not self.playback:
                continue  
            
            if self.tts_service == "openai":
                self.player_stream.write(audio_chunk)
            else:
                # For elevenlabs, ensure that the mpv process is available.
                if self.mpv_process is not None and self.mpv_process.stdin:
                    self.mpv_process.stdin.write(audio_chunk)
                    self.mpv_process.stdin.flush()

    def start_tts_thread(
        self, client: OpenAI | ElevenLabs
    ) -> threading.Thread:
        """
        Start a background thread that consumes text from the speech queue,
        converts it to speech using the TTS service, and streams the resulting audio.

        Args:
            client: An instance of the TTS service client.

        Returns:
            The thread object handling the TTS conversion.
        """

        def text_to_speech_worker() -> None:
            while True:
                sentence: str = self.speech_queue.get()
                if sentence is None:
                    break
                
                if self.playback:
                    self.text_to_speech(sentence, client)
                    self.speech_queue.task_done()

        tts_thread = threading.Thread(target=text_to_speech_worker, daemon=True)
        tts_thread.start()

        tts_voice_thread = threading.Thread(target=self.audio_stream_worker, daemon=True)
        tts_voice_thread.start()

        return tts_thread
