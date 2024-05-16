import time
import cv2
import openai
from openai import AssistantEventHandler
import os
from dotenv import load_dotenv
import wave
import pyaudio
from PIL import Image
import numpy as np
from typing_extensions import override
import simpleaudio as sa
import warnings


# Ignore DeprecationWarning (comment this out if you want to see the warnings!!)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables from .env file
load_dotenv()

# Audio recording parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100  # Sampling rate (samples per second)
CHUNK = 1024  # Number of frames per buffer
WAVE_RECORDING_FILENAME = "data/recordings/recording.wav"  # Name of the output file
SPEECH_FILENAME = "data/outputs/speech.wav"  # Name of the speech file
# Picture file parameters
IMAGE_FILENAME = "data/images/image"  # Name of the image file .png gets added later

# Set up OpenAI API credentials
client = openai.Client ()

###################
#assistant text stream
class EventHandler(AssistantEventHandler):    
  @override
  def on_text_created(self, text) -> None:
    print(f"\nassistant > ", end="", flush=True)
      
  @override
  def on_text_delta(self, delta, snapshot):
    print(delta.value, end="", flush=True)
      
  def on_tool_call_created(self, tool_call):
    print(f"\nassistant > {tool_call.type}\n", flush=True)
  
  def on_tool_call_delta(self, delta, snapshot):
    if delta.type == 'code_interpreter':
      if delta.code_interpreter.input:
        print(delta.code_interpreter.input, end="", flush=True)
      if delta.code_interpreter.outputs:
        print(f"\n\noutput >", flush=True)
        for output in delta.code_interpreter.outputs:
          if output.type == "logs":
            print(f"\n{output.logs}", flush=True)


# Function to record audio
def record_audio(duration):
    """
    Records audio for a specified duration and saves it as a WAV file.
    
    :param duration: Duration of recording in seconds.
    """
    audio = pyaudio.PyAudio()

    # Open the audio stream
    stream = audio.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Recording...")

    frames = []

    # Record audio in chunks
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

 # Save the recorded audio to a file
    with wave.open(WAVE_RECORDING_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# Function to capture an image
def capture_image(num_pictures, delay):

    print("Capturing images...")

    cap = cv2.VideoCapture(0)
    # Skip the first 5 frames to allow the camera to adjust to lighting
    for _ in range(5):
        cap.read()
    for i in range(num_pictures):
        ret, frame = cap.read()
        # Save image with different file names
        image_filename = f"{IMAGE_FILENAME}_{i+1}.png"
        cv2.imwrite(image_filename, frame)
        time.sleep(delay)
    cap.release()

    print("Images captured.")

# Function to send audio to OpenAI Whisper and get transcript
def transcribe_audio(file_path):
    """
    Transcribes an audio file using OpenAI's Whisper.
    
    :param file_path: Path to the audio file.
    :return: Transcription text.
    """
    try:

        audio_file = open(file_path, 'rb')
        transcription_text = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        return transcription_text.text

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        transcription_text = "THAT DIDN'T WORK."
        return transcription_text
    
#function to make image file for GPT
def make_image_file(image_path):
    """
    Convert an image file to a format that OpenAI's API can process.
    
    :param image_path: Path to the image file.
    :return: Image file in a format that OpenAI's API can process.
    """
    file = client.files.create(
        file=open(image_path, "rb"),
        purpose="vision"
    )
    return file    

# Function to add image to thread
def add_image_to_thread(thread_id, file_id):
    """
    Add an image to a thread.
    
    :param thread_id: ID of the thread.
    :param image_file: Image file to add.
    """
    message = client.beta.threads.messages.create(
        thread_id,
        content=[
            {
                "type": "image_file",
                "image_file": {"file_id": file_id}
            },
        ],
         role="user"    
        )

# Function to add text to thread
def add_text_to_thread(thread_id, text):
    """
    Add text to a thread.
    
    :param thread_id: ID of the thread.
    :param text: Text to add.
    """
    message = client.beta.threads.messages.create(
        thread_id,
        content=[
            {
                "type": "text",
                "text": text
            },
        ],
         role="user"    
        )

# Function to run the thread and stream the response
def run_thread_and_stream(thread_id, assistant_id):
    """
    Run the thread and stream the response.
    
    :param thread_id: ID of the thread.
    :param assistant_id: ID of the assistant.
    """
    with client.beta.threads.runs.stream(
            max_completion_tokens=400,
            thread_id=thread_id,
            assistant_id=assistant_id,
            event_handler=EventHandler(),
            ) as stream:
            stream.until_done()
    return stream

# Function to use text-to-speech to stream response to a file
def text_to_speech(speech_text):
    response    = client.audio.speech.create(
       model="tts-1",
       voice="alloy",
       response_format="wav",
       input=speech_text
    ) 
    response.stream_to_file(SPEECH_FILENAME)

# function to play wav file 
def play_wav(file_path):
    """
    Play a WAV file using the simpleaudio library.
    
    :param file_path: Path to the WAV file.
    """
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()   


# Main program
def main():

    # get assistant id
    assistant = client.beta.assistants.retrieve(os.environ.get("ASSISTANT_ID"))
    assistant_id = assistant.id

    #create a new thread
    thread = client.beta.threads.create()   
    thread_id = thread.id

    while True:
        input("Press Enter to start recording audio...")
        # Record audio for 5 seconds
        record_audio(5)

        # Capture an image
        capture_image(3, 0.5)

        # Send audio to OpenAI Whisper
        transcription = transcribe_audio(WAVE_RECORDING_FILENAME)

        # Display the transcription
        if transcription:
            print("Transcription:", transcription)
        else:
            print("Transcription failed.")

        # Get all image files in the /data/images directory
        image_files = os.listdir("data/images")

        #sort the list of image files
        image_files.sort()

        # Iterate over each image file
        for image_file in image_files:
            # Skip the .gitkeep file
            if image_file == ".gitkeep":
                continue

            # Create image file for OpenAI API
            image_path = os.path.join("data/images", image_file)
            print(">>>>>>>>>>>>>>>>>>>"+image_path)
            image_file = make_image_file(image_path)

            # Add image to the thread
            add_image_to_thread(thread_id, image_file.id)

        # Add text to the thread
        add_text_to_thread(thread_id, transcription)

        # Run thread and stream the answer
        while True:
            stream = run_thread_and_stream(thread_id, assistant_id)

            # Save the final messages
            if stream.on_message_done:
                final_message = stream.get_final_messages()
            if final_message:
                break

        #print text
        final_text = final_message[0].content[0].text.value

        # use tts to speak the response
        text_to_speech(final_text)
        play_wav(SPEECH_FILENAME)

        print("\n")
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()