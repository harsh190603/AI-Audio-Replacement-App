import os
import subprocess
import torch
import torchaudio
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from textblob import TextBlob
from TTS.api import TTS
import numpy as np
import json
import streamlit as st
import librosa
import language_tool_python  # Ensure this is imported
from spellchecker import SpellChecker  # Import PySpellChecker
import re

# Initialize LanguageTool and PySpellChecker
tool = language_tool_python.LanguageTool('en-US')
spell_checker = SpellChecker()

# Load MFA dictionary
def load_dictionary(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:  # Specify UTF-8 encoding
        dictionary = set(line.split()[0] for line in f if line.strip())
    return dictionary

def extract_audio(video_path, output_dir):
    video = VideoFileClip(video_path)
    audio_path = os.path.join(output_dir, "audio.wav")
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    audio = AudioSegment.from_wav(audio_path)
    mono_audio = audio.set_channels(1)
    mono_audio.export(audio_path, format='wav')
    return audio_path

def transcribe_audio(audio_path):
    model = Model("vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(model, 16000)
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    waveform = waveform.numpy()[0]
    waveform = (waveform * 32767).astype(np.int16)
    rec.AcceptWaveform(waveform.tobytes())
    result = json.loads(rec.Result())
    transcript = result.get('text', '')
    return transcript

def correct_grammar(transcript):
    words = transcript.split()  
    corrected_transcript = ' '.join([spell_checker.candidates(word).pop() if spell_checker.unknown([word]) else word for word in words])
    matches = tool.check(corrected_transcript)
    corrected_transcript = language_tool_python.utils.correct(corrected_transcript, matches)
    return corrected_transcript

def clean_transcription_file(corrected_transcript, dictionary):
    """Cleans a single transcription file by removing non-standard characters."""
    # with open(transcription_path, 'r', encoding='utf-8') as file:
    #     content = file.read()
    
    # Remove non-standard characters using a regex pattern
    cleaned_content = re.sub(r'[^a-zA-Z0-9\s,.?!\'"-]', '', corrected_transcript)  # Adjust regex as needed
    
    # Split cleaned content into words and filter only valid words present in the dictionary
    words = cleaned_content.split()
    filtered_words = [word for word in words if word in dictionary]
    
    # Reconstruct the filtered transcription and save it
    filtered_transcript = ' '.join(filtered_words)
    
    return filtered_transcript
    # with open(transcription_path, 'w', encoding='utf-8') as file:
    #     file.write(filtered_transcription)

def align_transcript(audio_path, filtered_transcript, output_dir):
    # Filter the corrected transcript to include only words in the dictionary
    # filtered_transcript = ' '.join([word for word in filtered_transcript.split() if word in dictionary])
    
    if not os.path.exists(audio_path):
        st.error(f"Audio file not found: {audio_path}")
        return None

    filtered_transcript_path = os.path.join(os.path.dirname(audio_path), 'audio.txt')
    with open(filtered_transcript_path, 'w') as f:
        f.write(filtered_transcript)  # Use the filtered transcript

    beam_size = 40
    retry_beam_size = 160
    mfa_command = f"mfa align C:/Users/Harsh/Documents/MFA/my_corpus C:/Users/Harsh/Documents/MFA/pretrained_models/dictionary/english_us_mfa.dict C:/Users/Harsh/Documents/MFA/pretrained_models/acoustic/english_mfa C:/Users/Harsh/Desktop/Video_Audio_Replacer_App/Montreal-Forced-Aligner/output --beam {beam_size} --retry_beam {retry_beam_size}"
    process = subprocess.run(mfa_command, shell=True, capture_output=True)
    if process.returncode != 0:
        st.error(f"MFA alignment failed: {process.stderr.decode()}")
        return None
    aligned_textgrid_path = os.path.join(output_dir, 'audio.wav').replace('.wav', '.TextGrid')
    return aligned_textgrid_path

def extract_word_intervals(textgrid_path):
    word_intervals = []
    
    with open(textgrid_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    in_words_tier = False
    for i, line in enumerate(lines):
        line = line.strip()
        
        if 'name = "words"' in line:
            in_words_tier = True
        
        if in_words_tier:
            if 'intervals [' in line:
                xmin_line = lines[i + 1].strip()
                xmax_line = lines[i + 2].strip()
                text_line = lines[i + 3].strip()
                
                xmin = float(xmin_line.split('=')[1].strip())
                xmax = float(xmax_line.split('=')[1].strip())
                text = text_line.split('=')[1].strip().replace('"', '')
                
                if text:
                    word_intervals.append((text, (xmin, xmax)))
                    
            if 'item [' in line and not 'class = "IntervalTier"' in line:
                break
    
    return word_intervals

def adjust_audio_speed(segment_path, desired_duration, actual_duration):
    try:
        if actual_duration > 0:
            speed_factor = actual_duration / desired_duration
            speed_factor = max(0.5, min(100, speed_factor))
            audio = AudioSegment.from_file(segment_path)
            output_path = "adjusted_audio_temp.wav"

            subprocess.run([
                "ffmpeg", 
                "-y",                           
                "-i", segment_path,            
                "-filter:a", f"atempo={speed_factor}", 
                "-c:v", "copy",                
                output_path                    
            ], check=True)

            adjusted_segment = AudioSegment.from_file(output_path)
            return adjusted_segment

    except ZeroDivisionError:
        print("ZeroDivisionError: Actual duration is zero.")
    except subprocess.CalledProcessError:
        print("Error occurred while processing the audio with ffmpeg.")
    
    return None

def generate_audio_with_durations(corrected_transcript, word_intervals):
    tts = TTS(model_name="tts_models/en/ljspeech/glow-tts")
    audio_segments = []

    for word, (start, end) in zip(corrected_transcript.split(), [interval[1] for interval in word_intervals]):
        audio = tts.tts(word)
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        folder_name = os.path.join("C:/Users/Harsh/Documents/MFA/my_corpus/utterances", str(int(start * 1000)))  # in milliseconds
        os.makedirs(folder_name, exist_ok=True)

        segment_path = os.path.join(folder_name, f"{word}.wav")
        torchaudio.save(segment_path, audio_tensor, sample_rate=16000)

        audio_segments.append((segment_path, float(start), float(end)))

    combined = AudioSegment.silent(duration=int((word_intervals[-1][1][1] * 1000)))  # duration in ms

    for (segment_path, start, end) in audio_segments:
        segment = AudioSegment.from_wav(segment_path)

        desired_duration = (end - start) * 1000  # Desired duration in milliseconds
        actual_duration = len(segment)  # Length of the segment in milliseconds
        
        adjusted_segment = adjust_audio_speed(segment_path, desired_duration, actual_duration)

        combined = combined.overlay(adjusted_segment, position=int(start * 1000))  # Position in milliseconds

    final_audio_path = os.path.join("C:/Users/Harsh/Documents/MFA/my_corpus", "final_generated_audio.wav")
    combined.export(final_audio_path, format="wav")
    return final_audio_path

def replace_audio_in_video_ffmpeg(video_file, new_audio_path, output_dir):
    final_video_path = os.path.join(output_dir, "final_video.mp4")
    command = [
        "ffmpeg",
        "-i", video_file,
        "-i", new_audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_video_path
    ]
    subprocess.run(command, check=True)
    return final_video_path

def main():
    # Streamlit interface
    st.title("Video Audio Replacement App")

    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        output_dir = "C:/Users/Harsh/Desktop/Video_Audio_Replacer_App/Montreal-Forced-Aligner/output"
        if st.button("Replace Audio"):
            video_path = os.path.join(output_dir, "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            audio_path = extract_audio(video_path, "C:/Users/Harsh/Documents/MFA/my_corpus")
            if not audio_path or not os.path.exists(audio_path):
                st.error("Audio extraction failed or audio file not found.")
            else:
                transcript = transcribe_audio(audio_path)
                st.write(f"Original Transcript:-\n {transcript}")
                dictionary_path = "C:/Users/Harsh/Documents/MFA/pretrained_models/dictionary/english_us_mfa.dict"
                dictionary = load_dictionary(dictionary_path)  # Load the MFA dictionary
                corrected_transcript = correct_grammar(transcript)
                st.write(f"Corrected Transcript:-\n {corrected_transcript}")
                
                # Clean transcription file using the dictionary
                corrected_transcription_path = os.path.join(output_dir, "corrected_transcription.txt")
                with open(corrected_transcription_path, 'w', encoding='utf-8') as f:
                    f.write(corrected_transcript)

                filtered_transcript=clean_transcription_file(corrected_transcript, dictionary)
                st.write(f"Filtered Transcript:-\n {filtered_transcript}")

                filtered_transcription_path = os.path.join(output_dir, "filtered_transcription.txt")
                with open(filtered_transcription_path, 'w', encoding='utf-8') as f:
                    f.write(filtered_transcript)

                aligned_textgrid_path = align_transcript(audio_path, filtered_transcript, output_dir)

                if aligned_textgrid_path:
                    word_intervals = extract_word_intervals(aligned_textgrid_path)
                    final_audio_path = generate_audio_with_durations(corrected_transcript, word_intervals)
                    final_video_path=replace_audio_in_video_ffmpeg(video_path, final_audio_path, output_dir)
                    st.success(f"Final video saved at: {final_video_path}")
                else:
                    st.success("Audio replaced successfully!")
                
            # Display the uploaded video
            st.video(video_path, format="video/mp4", start_time=0)
            # Display the output video (assuming post-processed)
            if os.path.exists(final_video_path):
                st.subheader("Output Video")
                st.video(final_video_path, format="video/mp4", start_time=0)

if __name__ == "__main__":
    main()
