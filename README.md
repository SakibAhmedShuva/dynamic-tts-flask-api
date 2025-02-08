# Dynamic TTS Flask API

A robust Text-to-Speech (TTS) API service built with Flask and PyTorch, leveraging the MMS-TTS model from Facebook. This service provides multiple voice options, real-time audio streaming, and various output formats.

## Features

- Multiple voice options (male and female variants)
- Real-time audio streaming
- Support for multiple output formats (WAV, FLAC, OGG, MP3)
- Voice modification capabilities (pitch and speed adjustment)
- Asynchronous task processing
- RESTful API endpoints
- Built-in task status monitoring

## Prerequisites

```bash
pip install flask torch transformers soundfile pydub numpy
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/SakibAhmedShuva/dynamic-tts-flask-api.git
cd dynamic-tts-flask-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python tts.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### 1. Convert Text to Speech
```http
POST /api/tts/convert
```

Request body:
```json
{
    "text": "Your text to convert",
    "voice": "female_1",
    "language": "en-US"
}
```

Response:
```json
{
    "task_id": "task_20240208_123456",
    "status": "processing"
}
```

### 2. Check Conversion Status
```http
GET /api/tts/status/<task_id>
```

Response:
```json
{
    "status": "completed",
    "audio_path": "tts_outputs/task_20240208_123456.wav"
}
```

### 3. Save Speech File
```http
POST /api/tts/save
```

Request body:
```json
{
    "text": "Your text to convert",
    "voice": "female_1",
    "language": "en-US",
    "format": "wav"
}
```

Response:
```json
{
    "status": "success",
    "file_path": "tts_outputs/speech_20240208_123456.wav"
}
```

### 4. Get Available Voices
```http
GET /api/tts/voices
```

Response:
```json
{
    "voices": [
        {
            "id": "female_1",
            "language": "en-US",
            "gender": "female",
            "description": "Clear and professional female voice"
        },
        // ... other voices
    ]
}
```

### 5. Stream Speech
```http
POST /api/tts/stream
```

Request body:
```json
{
    "text": "Your text to convert",
    "voice": "female_1",
    "language": "en-US"
}
```

Returns: Audio stream (audio/wav)

### 6. Download Audio
```http
GET /api/tts/download/<task_id>
```

Returns: Audio file (audio/wav)

## Voice Options

- `female_1`: Clear and professional female voice
- `female_2`: Warm and friendly female voice
- `male_1`: Deep and confident male voice
- `male_2`: Gentle and calm male voice

## Supported Audio Formats

- WAV
- FLAC
- OGG
- MP3

## Configuration

Voice configurations can be modified in the `PublicDynamicTTS` class:

```python
self.voices = {
    "female_1": {"pitch_shift": 1.1, "speed": 1.0},
    "female_2": {"pitch_shift": 1.2, "speed": 0.95},
    "male_1": {"pitch_shift": 0.9, "speed": 1.0},
    "male_2": {"pitch_shift": 0.8, "speed": 0.95}
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request (missing required parameters)
- 404: Not Found (audio file or task not found)
- 500: Internal Server Error (conversion or processing failed)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook MMS-TTS model
- Flask framework
- PyTorch
- Transformers library

## Contact

Project Link: [https://github.com/SakibAhmedShuva/dynamic-tts-flask-api](https://github.com/SakibAhmedShuva/dynamic-tts-flask-api)
