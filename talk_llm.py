import argparse
import os
import re
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


def preload_ollama(llm_model, console):
    with console.status(f"[bold green]Loading LLM: {llm_model}...") as status:
        ollama.chat(model=llm_model, messages=[{"role": "user", "content": "reply with 'yes'"}])

def init_console():
    console = Console()
    console.clear()
    return console

def load_whisper(model_name, console):
    with console.status("[bold green]Loading Whisper model...") as status:
        model = whisper.load_model(model_name)
    return model

def initialize_text_to_speech(p):
    player_stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    return player_stream

def start_tts_thread(speech_queue, client, p):
    def text_to_speech_worker():
        while True:
            sentence = speech_queue.get()
            if sentence is None:
                break
            text_to_speech(sentence, speech_queue, client, p)
            speech_queue.task_done()

    tts_thread = threading.Thread(target=text_to_speech_worker)
    tts_thread.start()
    return tts_thread

def text_to_speech(llm_response, speech_queue, client, p):
    player_stream = initialize_text_to_speech(p)
    with client.audio.speech.with_streaming_response.create( 
         model="tts-1", 
         voice="echo",  # "alloy", "echo", "fable", "onyx", "shimmer",
         speed=1,
         response_format="pcm",  # similar to WAV, but without a header chunk at the start. 
         input=llm_response
    ) as response: 
        for chunk in response.iter_bytes(chunk_size=1024): 
            if keyboard.is_pressed('s'):  # Stop playback if 's' is pressed
                speech_queue.queue.clear()
                break
            player_stream.write(chunk)

def stream_llm_response(llm_model, messages, speech_queue, console):

    stream = ollama.chat(
        model = llm_model,
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
                    speech_queue.put(line.strip())
                else:
                    current_text = line
            else:
                # Last element, might be incomplete
                current_text = line

    print("\n")
    # Process any remaining text after streaming
    if current_text.strip():
        speech_queue.put(current_text.strip())

    # Append the entire response to messages once complete
    messages.append({
        "role": "assistant",
        "content": full_text
    })

    
def transcribe_audio(file_path, model, messages, console):
    if file_path:
        with console.status("[bold green]transcribing...") as status:
            result = model.transcribe(file_path)
            transcript = result['text']  # Store the transcript in a local variable
            console.print("[green]User: [yellow]" + result['text'])

            messages.append({
                "role": "user",
                "content": transcript
             })
    else:
        console.print("[red]No audio file provided for transcription.")

    os.remove(file_path)
    return result['text']

def record_audio(speech_queue, console):
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

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--whisper", type=str, default="large-v3", help="The name of the Whisper model to use.")
    argparser.add_argument("--llm_model", type=str, default="jarvis-english", help="The name of the LLM model to use.")
    
    args = argparser.parse_args()

    messages = []
    console = Console()
    console.clear()
    model = load_whisper(model_name=args.whisper, console=console)
    client = OpenAI()
    p = pyaudio.PyAudio()
    speech_queue = queue.Queue()
    
    preload_ollama(llm_model=args.llm_model, console=console)


    tts_thread = start_tts_thread(speech_queue, client, p)

    while True:
        file_path = record_audio(speech_queue, console)
        transcript = transcribe_audio(file_path, model, messages, console)
        stream_llm_response(args.llm_model, messages, speech_queue, console)

if __name__ == "__main__":
    main_loop()