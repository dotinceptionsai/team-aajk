import asyncio
import base64
import json

import pyaudio
import websockets

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

TRANSCRIPT_SERVICE_URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"


class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )
        print(self.p.get_default_input_device_info())

    def terminate(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def next_data(self):
        data = self.stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        data = base64.b64encode(data).decode("utf-8")
        return data


class AudioTranscriber:
    def __init__(self, api_key_assembly_ai: str):
        if api_key_assembly_ai is None:
            raise ValueError("API key for AssemblyAI is required")
        self.audio_recorder = AudioRecorder()
        self.api_key_assembly_ai = api_key_assembly_ai

    def subscribe(self, callback):
        self.callback = callback

    async def send_receive(self):
        print(f"Connecting websocket to url ${TRANSCRIPT_SERVICE_URL}")
        async with websockets.connect(
            TRANSCRIPT_SERVICE_URL,
            extra_headers=(("Authorization", self.api_key_assembly_ai),),
            ping_interval=5,
            ping_timeout=20,
        ) as _ws:
            await asyncio.sleep(0.1)
            print("Receiving SessionBegins ...")
            session_begins = await _ws.recv()
            print(session_begins)
            print("Sending messages ...")

            async def send():
                while True:
                    try:
                        data = self.audio_recorder.next_data()
                        json_data = json.dumps({"audio_data": str(data)})
                        await _ws.send(json_data)
                    except websockets.exceptions.ConnectionClosedError as e:
                        print(e)
                        assert e.code == 4008
                        break
                    except Exception as e:
                        assert False, "Not a websocket 4008 error"
                    await asyncio.sleep(0.01)
                return True

            async def receive():
                while True:
                    try:
                        result_str = await _ws.recv()
                        result = json.loads(result_str)
                        text = result["text"]
                        if text and result["message_type"] == "FinalTranscript":
                            await self.callback(text)
                    except websockets.exceptions.ConnectionClosedError as e:
                        print(e)
                        assert e.code == 4008
                        break
                    except Exception as e:
                        print(e)
                        assert False, "Not a websocket 4008 error"

            await asyncio.gather(send(), receive())

    async def start(self):
        await self.send_receive()

    def stop(self):
        self.audio_recorder.stop()
