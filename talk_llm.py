import argparse
import os
import re
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import torch
import whisper
import threading
import keyboard
from rich.console import Console
from rich.spinner import Spinner
from rich.markdown import Markdown
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import ollama
from openai import OpenAI
import pyaudio
import queue
from faster_whisper import WhisperModel

from audio import AudioManager

def preload_ollama(llm_model, console: Console):
    with console.status(f"[bold green]Loading LLM: {llm_model}...") as status:
        model_list = ollama.list()

        if llm_model not in model_list.keys():
            ollama.pull(llm_model)
        
        ollama.chat(model=llm_model, messages=[{"role": "user", "content": "reply with 'yes'"}])

def init_console():
    console = Console()
    console.clear()
    return console

def load_whisper(model_name: str, console: Console):
    with console.status("[bold green]Loading Whisper model...") as status:
        model = whisper.load_model(model_name)
    return model


def stream_llm_response(llm_model: str, messages: list[str], speech_queue: queue.Queue, silent_flag: bool, console: Console):

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
    
        lines = re.split(r'(?<!\.\.)\.(?!\.)|(?<=[!?])\s+', current_text)
    
        if not silent_flag:
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    # Check if it's a list item or an incomplete sentence
                    if not re.match(r"^\d+\.", line.strip()):
                        speech_queue.put(line.strip())
                    else:
                        current_text = line
                else:
                    # Last element, might be incomplete
                    current_text = line

    print("\n")
    if not silent_flag:  
        # Process any remaining text after streaming
        if current_text.strip():
            speech_queue.put(current_text.strip())

    # Append the entire response to messages once complete
    messages.append({
        "role": "assistant",
        "content": full_text
    })

def transcribe_audio(file_path: str, model: WhisperModel, messages: list[str], console: Console, context: str | None) -> str:
    if file_path:
        with console.status("[bold green]transcribing...") as status:
            segments, info = model.transcribe(file_path)
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            if context:
                content = "Context: " + context + "\n\n User: " + transcription 
                console.print("[green]Context: [light_slate_blue]" + context)
            else:
                content = transcription

            console.print("[green]User: [yellow]" + transcription)

            messages.append({
                "role": "user",
                "content": content
             })
    else:
        console.print("[red]No audio file provided for transcription.")

    os.remove(file_path)
    return transcription

def main_loop():

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--whisper", type=str, default="small", help="The name of the Whisper model to use.")
    argparser.add_argument("--llm", type=str, default="llama3", help="The name of the LLM model to use.")
    argparser.add_argument("--silent", action="store_true", help="Disable TTS.")
    argparser.add_argument("--tts", choices=["openai", "elevenlabs"], default="openai", help="The TTS service to use.")
    argparser.add_argument("--lang", type=str, choices=["en", "multi"], default="en", help="The language to use for the TTS service.")
    argparser.add_argument("--voice_id", type=str, default="ErXwobaYiN019PkySvjV", help="The voice ID to use for Eleven Labs TTS.")


    args = argparser.parse_args()
    messages = []
    console = Console()

    console.clear()


    # check if cuda is available
    if not torch.cuda.is_available():
        console.print("[red]CUDA is not available. Using CPU for inference.")
        model = WhisperModel(args.whisper, device="cpu", compute_type="float32")
    else:
        model = WhisperModel(args.whisper, device="cuda", compute_type="float16")

    client = OpenAI()

    if args.tts == "elevenlabs":
        eleven_client = ElevenLabs()

    speech_queue = queue.Queue()
    preload_ollama(llm_model=args.llm, console=console)

    audio_manager = AudioManager(speech_queue, console, args.tts, args.lang, args.voice_id)
    tts_thread = audio_manager.start_tts_thread(eleven_client if args.tts == "elevenlabs" else client)

    while True:
        file_path, context = audio_manager.record_audio()

        if file_path:
            transcribe_audio(file_path, model, messages, console, context)
            stream_llm_response(args.llm, messages, speech_queue, args.silent, console)

if __name__ == "__main__":
    main_loop()