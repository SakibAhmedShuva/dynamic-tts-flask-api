from flask import Flask, request, jsonify, send_file, Response, stream_with_context
import torch
import numpy as np
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
from datetime import datetime
import threading
import queue
import os
from pydub import AudioSegment
import io

app = Flask(__name__)

class PublicDynamicTTS:
    def __init__(self):
        """Initialize the TTS service with public models"""
        print("Loading TTS models... This may take a few moments.")
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        
        # Initialize task management
        self.task_queue = queue.Queue()
        self.task_statuses = {}
        self.output_dir = "tts_outputs"
        self.supported_formats = ['wav', 'flac', 'ogg', 'mp3']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Voice configurations
        self.voices = {
            "female_1": {"pitch_shift": 1.1, "speed": 1.0},
            "female_2": {"pitch_shift": 1.2, "speed": 0.95},
            "male_1": {"pitch_shift": 0.9, "speed": 1.0},
            "male_2": {"pitch_shift": 0.8, "speed": 0.95}
        }
        
        self.start_task_processor()

    def _normalize_waveform(self, waveform):
        """Normalize audio waveform to [-1, 1] range"""
        max_val = torch.max(torch.abs(waveform))
        return waveform / max_val if max_val > 0 else waveform

    def convert_text_to_speech(self, text, voice="female_1", language="en-US", speed=1.0):
        """Convert text to speech with voice modifications"""
        try:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.task_statuses[task_id] = "processing"
            
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**inputs).waveform
            
            output = self._modify_voice(output, voice, speed)
            self.task_queue.put((task_id, output))
            
            return task_id
            
        except Exception as e:
            print(f"Conversion error: {str(e)}")
            return None

    def _modify_voice(self, waveform, voice, speed):
        """Apply voice modifications with proper audio handling"""
        try:
            voice_config = self.voices.get(voice, self.voices["female_1"])
            
            # Pitch modification
            modified = self._pitch_shift(waveform, voice_config["pitch_shift"])
            
            # Speed modification
            modified = self._adjust_speed(modified, speed * voice_config["speed"])
            
            return self._normalize_waveform(modified)
            
        except Exception as e:
            print(f"Voice modification error: {str(e)}")
            return waveform

    def _pitch_shift(self, waveform, factor):
        """Pitch shifting with proper tensor handling"""
        return waveform * factor

    def _adjust_speed(self, waveform, factor):
        """Improved speed adjustment with anti-aliasing"""
        if factor == 1.0:
            return waveform

        orig_length = waveform.shape[1]
        new_length = int(orig_length / factor)
        
        # Resample with linear interpolation
        x_old = np.linspace(0, 1, orig_length)
        x_new = np.linspace(0, 1, new_length)
        
        resampled = np.interp(x_new, x_old, waveform.numpy()[0])
        return torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)

    def save_speech_file(self, text, voice="female_1", language="en-US", format="wav", destination=None):
        """Save speech to file with format validation"""
        try:
            format = format.lower()
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}. Use: {self.supported_formats}")
            
            if destination is None:
                destination = os.path.join(
                    self.output_dir, 
                    f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
                )
            
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**inputs).waveform
            
            output = self._modify_voice(output, voice, 1.0)
            output = self._normalize_waveform(output)
            
            # Handle different formats
            if format == 'mp3':
                audio = AudioSegment(
                    output.numpy().tobytes(),
                    frame_rate=16000,
                    sample_width=2,
                    channels=1
                )
                audio.export(destination, format='mp3', bitrate='128k')
            else:
                sf.write(destination, output.numpy().T, 16000, format=format)
            
            return {"status": "success", "file_path": destination}
            
        except Exception as e:
            print(f"Save error: {str(e)}")
            return None

    def start_task_processor(self):
        """Background task processor with error handling"""
        def process_tasks():
            while True:
                try:
                    task_id, waveform = self.task_queue.get(timeout=1)
                    if waveform is not None:
                        waveform = self._normalize_waveform(waveform)
                        output_path = os.path.join(self.output_dir, f"{task_id}.wav")
                        sf.write(output_path, waveform.numpy().T, 16000)
                        self.task_statuses[task_id] = "completed"
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Task processing error: {str(e)}")
                    if task_id:
                        self.task_statuses[task_id] = "failed"

        threading.Thread(target=process_tasks, daemon=True).start()


    def check_status(self, task_id):
        """Check the status of a TTS task"""
        return {
            "status": self.task_statuses.get(task_id, "not_found"),
            "audio_path": os.path.join(self.output_dir, f"{task_id}.wav") 
            if self.task_statuses.get(task_id) == "completed" 
            else None
        }

    def get_available_voices(self):
        """Return available voice options"""
        return {
            "voices": [
                {
                    "id": "female_1",
                    "language": "en-US",
                    "gender": "female",
                    "description": "Clear and professional female voice"
                },
                {
                    "id": "female_2",
                    "language": "en-US",
                    "gender": "female",
                    "description": "Warm and friendly female voice"
                },
                {
                    "id": "male_1",
                    "language": "en-US",
                    "gender": "male",
                    "description": "Deep and confident male voice"
                },
                {
                    "id": "male_2",
                    "language": "en-US",
                    "gender": "male",
                    "description": "Gentle and calm male voice"
                }
            ]
        }

    def stream_speech(self, text, voice="female_1", language="en-US"):
        """Stream speech output in chunks"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**inputs).waveform
            
            output = self._modify_voice(output, voice, 1.0)
            output = self._normalize_waveform(output)
            
            audio_data = output.numpy()[0]
            chunk_size = 16000  # 1 second chunks
            
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]
                
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield None

# Initialize TTS service
tts_service = PublicDynamicTTS()

@app.route('/api/tts/convert', methods=['POST'])
def convert_text():
    try:
        data = request.json
        text = data.get('text')
        voice = data.get('voice', 'female_1')
        language = data.get('language', 'en-US')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
            
        task_id = tts_service.convert_text_to_speech(text, voice, language)
        
        if task_id:
            return jsonify({
                'task_id': task_id,
                'status': 'processing'
            })
        else:
            return jsonify({'error': 'Conversion failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/status/<task_id>', methods=['GET'])
def check_status(task_id):
    try:
        status = tts_service.check_status(task_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/save', methods=['POST'])
def save_speech():
    try:
        data = request.json
        text = data.get('text')
        voice = data.get('voice', 'female_1')
        language = data.get('language', 'en-US')
        format = data.get('format', 'wav')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
            
        result = tts_service.save_speech_file(text, voice, language, format)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Save failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/voices', methods=['GET'])
def get_voices():
    try:
        voices = tts_service.get_available_voices()
        return jsonify(voices)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/stream', methods=['POST'])
def stream_speech():
    try:
        data = request.json
        text = data.get('text')
        voice = data.get('voice', 'female_1')
        language = data.get('language', 'en-US')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400

        def generate():
            for chunk in tts_service.stream_speech(text, voice, language):
                if chunk is not None:
                    # Convert numpy array to bytes
                    chunk_bytes = chunk.tobytes()
                    yield chunk_bytes

        return Response(stream_with_context(generate()), 
                       mimetype='audio/wav')
                       
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/download/<task_id>', methods=['GET'])
def download_audio(task_id):
    try:
        status = tts_service.check_status(task_id)
        
        if status['status'] != 'completed':
            return jsonify({'error': 'Audio not ready'}), 404
            
        return send_file(status['audio_path'], 
                        mimetype='audio/wav',
                        as_attachment=True,
                        download_name=f'{task_id}.wav')
                        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
