from asyncio import sleep
import asyncio
from io import BytesIO
import pyaudio
import keyboard
import os
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import threading
from rich.console import Console
from rich.spinner import Spinner
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from openai import OpenAI
import queue



class AudioManager:
    def __init__(self, speech_queue, console, tts_service):
        self.tts_service = tts_service
        self.audio_stream_queue = queue.Queue()
        self.speech_queue = speech_queue
        self.console = console
        self.fs = 44100
        self.p = pyaudio.PyAudio()
        self.player_stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True, frames_per_buffer=8048)

    def record_audio(self):
        self.console.print("[blue]Press 'space' to start and 'enter' to stop the recording. Press 's' to stop TTS and 'q' to exit the program.")
        frames = []
        flag = False

        def callback(indata, frame_count, time_info, status):
            if flag:
                frames.append(indata.copy())

        with sd.InputStream(samplerate=self.fs, channels=1, callback=callback):
            while True:
                if keyboard.is_pressed('space'):
                    flag = True
                    break

                if keyboard.is_pressed('q'):
                    self.console.print("[red]Exiting...")
                    self.speech_queue.queue.clear()
                    os._exit(0)

                if keyboard.is_pressed('c'):
                    self.console.clear()
                    self.console.print("[blue]Press 'space' to start and 'enter' to stop the recording. Press 's' to stop TTS and 'q' to exit the program.")

                sd.sleep(100)

            with self.console.status("[bold green]recording...") as status:
                while True:
                    if keyboard.is_pressed('enter'):
                        flag = False
                        break
            sd.sleep(100)

        recording = np.vstack(frames)
        temp_file = tempfile.mktemp(suffix='.wav')
        sf.write(temp_file, recording, self.fs)

        return temp_file

    def text_to_speech(self, llm_response, client: OpenAI | ElevenLabs):
        buffer = []
        if self.tts_service == "openai":
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="echo",
                speed=1,
                response_format="pcm",
                input=llm_response
            ) as response:
                for chunk in response.iter_bytes(chunk_size=1024):
                    buffer.append(chunk)
                self.audio_stream_queue.put(buffer)

        elif self.tts_service == "elevenlabs":
            audio_stream = client.generate(
                optimize_streaming_latency="1",
                model="eleven_turbo_v2",
                voice="XRlny9TzSxQhHzOusWWe",
                output_format="pcm_24000",
                text=llm_response
            )
            for chunk in audio_stream:
                buffer.append(chunk) 
            self.audio_stream_queue.put(buffer)

    def audio_stream_worker(self):
        while True:
            audio_stream = self.audio_stream_queue.get()
            if audio_stream is None:
                break
            else:
                for audio_chunk in audio_stream:
                    if keyboard.is_pressed('s'):
                        break
                    silence = np.zeros(4096, dtype=np.int16)
                    self.player_stream.write(audio_chunk)
                self.player_stream.write(silence)
            self.audio_stream_queue.task_done()


    def start_tts_thread(self, client):
        def text_to_speech_worker():
            while True:
                sentence = self.speech_queue.get()
                if sentence is None:
                    break
                self.text_to_speech(sentence, client)
                self.speech_queue.task_done()

        tts_thread = threading.Thread(target=text_to_speech_worker)
        tts_voice_thread = threading.Thread(target=self.audio_stream_worker)
        tts_thread.start()
        tts_voice_thread.start()
        return tts_thread
