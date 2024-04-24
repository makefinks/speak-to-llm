from asyncio import sleep
import os
import re
import sys
import time
import wave
import requests
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import whisper
import threading
import keyboard
from rich.console import Console
from rich.spinner import Spinner
from rich.markdown import Markdown

import ollama
from openai import OpenAI
import pyaudio
import queue

# Initialize Colorama
console = Console()
console.clear()
# Load Whisper model
with console.status("[bold green]Loading Whisper model...") as status:
    model = whisper.load_model("large-v3")
client = OpenAI()

p = pyaudio.PyAudio()
speech_queue = queue.Queue()
messages = []

def text_to_speech_worker():
    while True:
        # Retrieve the next sentence from the queue; block if the queue is empty
        sentence = speech_queue.get()
        if sentence is None:
            # None is used as a signal to stop the worker thread
            break
        text_to_speech(sentence)
        speech_queue.task_done()

# Start the text-to-speech worker thread
tts_thread = threading.Thread(target=text_to_speech_worker)
tts_thread.start()

def text_to_speech(llm_response):
    player_stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True) 
    with client.audio.speech.with_streaming_response.create( 
         model="tts-1", 
         voice="onyx",  # "alloy", "echo", "fable", "onyx", "shimmer",
         speed=1.1,
         response_format="pcm",  # similar to WAV, but without a header chunk at the start. 
         input=llm_response
    ) as response: 
        for chunk in response.iter_bytes(chunk_size=1024): 
            if keyboard.is_pressed('s'):  # Stop playback if 's' is pressed
                speech_queue.queue.clear()
                break
            player_stream.write(chunk)
    player_stream.close()

def enqueue_text_to_speech(sentence):
    # Add a sentence to the queue
    speech_queue.put(sentence)

def stream_llm_response(transcript):
    messages.append({
        "role": "user",
        "content": transcript
    })

    stream = ollama.chat(
        model = "jarvis-english",
        messages=messages,
        stream=True
    )

    full_text = ""
    current_text = ""
    for chunk in stream:

        chunk_content = chunk['message']['content']
        full_text += chunk_content
        current_text += chunk_content
        console.print(chunk['message']['content'], end="")
    
        lines = re.split(r'(?<=[.!?])\s+', current_text)

        for i, line in enumerate(lines):
            if i < len(lines) - 1:  # Not the last element, hence complete
                # Check if it's a list item or an incomplete sentence
                if not re.match(r"^\d+\.", line.strip()):
                    enqueue_text_to_speech(line.strip())
                else:
                    current_text = line
            else:
                # Last element, might be incomplete
                current_text = line

    print("\n")
    # Process any remaining text after streaming
    if current_text.strip():
        enqueue_text_to_speech(current_text.strip())

    # Append the entire response to messages once complete
    messages.append({
        "role": "assistant",
        "content": full_text
    })

    
def transcribe_audio(file_path):
    if file_path:
        with console.status("[bold green]transcribing...") as status:
            result = model.transcribe(file_path)
            transcript = result['text']  # Store the transcript in a local variable
            console.print("[green]User: [yellow]" + result['text'])
    else:
        console.print("[red]No audio file provided for transcription.")

    os.remove(file_path)
    stream_llm_response(result['text'])

def record_audio():
    console.print("[blue]Press 'space' to start and 'enter' to stop the recording. Press 's' to stop TTS and 'q' to exit the program.")
    fs = 44100  # Sample rate
    frames = []  # List to hold audio frames
    flag = False

    def callback(indata, frame_count, time_info, status):
        if flag:
            frames.append(indata.copy())

    # Open a stream that uses callback for each block of data read
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        while True:
            if keyboard.is_pressed('space'):
                flag = True
                break
            
            if keyboard.is_pressed('q'):
                console.print("[red]Exiting...")
                speech_queue.queue.clear()
                os._exit(0)

            if keyboard.is_pressed('c'):
                console.clear()
                console.print("[blue]Press 'space' to start and 'enter' to stop the recording. Press 's' to stop TTS and 'q' to exit the program.")

            sd.sleep(100) 

        with console.status("[bold green]recording...") as status:
            while True:
                if keyboard.is_pressed('enter'):  # Stop recording when 'esc' is pressed
                    flag = False
                    break
        sd.sleep(100)
    recording = np.vstack(frames)  # Stack all the frames together
    temp_file = tempfile.mktemp(suffix='.wav')
    sf.write(temp_file, recording, fs)
    return temp_file

def main_loop():
    while True:
        file_path = record_audio()
        transcribe_audio(file_path)

# Run the main function
main_loop()
