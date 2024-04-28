from asyncio import sleep
import asyncio
from io import BytesIO
import subprocess
from typing import Literal
import pyaudio
from pynput.keyboard import Listener, Key
import os
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import threading
from rich.console import Console
from rich.spinner import Spinner
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings, stream
from openai import OpenAI
import queue
import time



class AudioManager:
    def __init__(self, speech_queue: queue.Queue, console: Console, tts_service: Literal["openai", "elevelabs"], language: Literal["en", "multi"], voice_id: str):
        self.tts_service = tts_service
        self.audio_stream_queue= queue.Queue()
        self.speech_queue = speech_queue
        self.console = console
        self.language = language
        self.voice_id = voice_id
        self.fs = 44100
        self.frames = []
        self.recording = False
        self.playback = True
        self.p = pyaudio.PyAudio()
        self.player_stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True, frames_per_buffer=8048)

        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        if tts_service == "elevenlabs":
            self.mpv_process = subprocess.Popen(
                mpv_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def on_press(self, key):
        if key == Key.space:
            self.recording = True
        elif key == Key.enter:
            if self.recording:
                self.recording = False
                self.playback = True
                return False  # Stop listener
        else:
            try:
                if key.char == 'q':
                    os._exit(0)
                elif key.char == "s":
                    if self.audio_stream_queue.qsize() > 0:
                        self.console.print("[bold red]Stopping playback...")
                        self.playback = False
                        self.speech_queue.queue.clear()
                        self.audio_stream_queue.queue.clear()
                elif key.char == "c":
                    self.console.clear()
            except AttributeError:
                # Handle the case where key.char is not available
                pass

    def record_audio(self):
        # Start listening to keyboard events in a separate thread to avoid blocking
        listener = Listener(on_press=self.on_press)
        listener.start()

        try:
            # Use the status indicator in a separate thread or manage it dynamically
            with self.console.status("[bold blue]Press 'space' to start and 'enter' to stop the recording. Press 's' to stop TTS and 'q' to exit the program.") as status:
                while not self.recording:
                    time.sleep(0.1)  # Small sleep to reduce CPU usage
                    if not listener.is_alive():
                        break  # If listener stops, break the loop

                if self.recording:
                    status.update("[bold green]Recording...")
                    with sd.InputStream(samplerate=self.fs, channels=1, callback=self.callback):
                        while self.recording:
                            time.sleep(0.1)  # Continue to sleep while recording to reduce CPU usage

        finally:
            listener.stop()
            listener.join()

        # After recording stops
        if not self.frames:
            self.console.print("[bold red]No audio recorded.")
            return None

        # Create the recording file
        recording = np.vstack(self.frames)
        temp_file = tempfile.mktemp(suffix='.wav')
        sf.write(temp_file, recording, self.fs)
        self.frames = []  # Clear frames after saving to file
        return temp_file

    def callback(self, indata, frame_count, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())


    def text_to_speech(self, llm_response, client: OpenAI | ElevenLabs):
        if self.tts_service == "openai":
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="echo",
                speed=1,
                response_format="pcm",
                input=llm_response
            ) as response:
                for chunk in response.iter_bytes(chunk_size=1024):
                    self.audio_stream_queue.put(chunk)
            # append silence because Openai voice starts instantly
            silence = np.zeros(4096, dtype=np.int16)
            self.audio_stream_queue.put(silence)

        elif self.tts_service == "elevenlabs":
            audio_stream = client.generate(
                optimize_streaming_latency="2",
                model="eleven_turbo_v2" if self.language == "en" else "eleven_multilingual_v2",
                voice=self.voice_id,
                voice_settings=VoiceSettings(
                    stability=0.6, similarity_boost=0.8, style=0.3, use_speaker_boost=True
                ),
                text=llm_response,
                stream=True
            )
            for chunk in audio_stream:
                self.audio_stream_queue.put(chunk)
            """ silence = np.zeros(4096, dtype=np.int16)
            self.audio_stream_queue.put(silence) """

    def audio_stream_worker(self):
        while True:
            if self.playback:
                if self.tts_service == "openai":
                    audio_chunk = self.audio_stream_queue.get()
                    if audio_chunk is None:
                        break
                
                    self.player_stream.write(audio_chunk)
                else:
                    audio_chunk = self.audio_stream_queue.get() 
                    if audio_chunk is None:
                        break

                    self.mpv_process.stdin.write(audio_chunk)  # type: ignore
                    self.mpv_process.stdin.flush()


    def start_tts_thread(self, client):
        def text_to_speech_worker():
            while True:
                sentence = self.speech_queue.get()
                if sentence is None:
                    break
                self.text_to_speech(sentence, client)
                self.speech_queue.task_done()

        tts_thread = threading.Thread(target=text_to_speech_worker)
        tts_thread.start()
        tts_voice_thread = threading.Thread(target=self.audio_stream_worker)
        tts_voice_thread.start()
       
        return tts_thread
