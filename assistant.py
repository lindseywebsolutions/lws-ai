import asyncio
import logging
import os
from dotenv import load_dotenv
from typing import Any, Optional
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("livekit-agent")

class AssistantFnc(llm.FunctionContext):
    def __init__(self, chat_ctx: Any) -> None:
        super().__init__()
        self.room: Any = None
        self.latest_video_frame: Any = None
        self.chat_ctx: Any = chat_ctx

    async def process_video_stream(self, track):
        """Process video stream and store the first video frame."""
        logger.info(f"Starting to process video track: {track.sid}")
        video_stream = rtc.VideoStream(track)
        try:
            async for frame_event in video_stream:
                self.latest_video_frame = frame_event.frame
                logger.info(f"Received a frame from track {track.sid}")
                break  # Process only the first frame
        except Exception as e:
            logger.error(f"Error processing video stream: {e}")

    @llm.ai_callable()
    async def capture_and_add_image(self) -> str:
        """Capture an image from the video stream and add it to the chat context."""
        if self.chat_ctx is None:
            logger.error("chat_ctx is not set")
            return "Error: chat_ctx is not set"

        video_publication = self._get_video_publication()
        if not video_publication:
            logger.info("No video track available")
            return "No video track available"

        try:
            await self._subscribe_and_capture_frame(video_publication)
            if not self.latest_video_frame:
                logger.info("No video frame available")
                return "No video frame available"

            chat_image = llm.ChatImage(image=self.latest_video_frame)
            self.chat_ctx.append(images=[chat_image], role="user")
            return f"Image captured and added to context. Dimensions: {self.latest_video_frame.width}x{self.latest_video_frame.height}"
        except Exception as e:
            logger.error(f"Error in capture_and_add_image: {e}")
            return f"Error: {e}"
        finally:
            self._unsubscribe_from_video(video_publication)
            self.latest_video_frame = None

    def _get_video_publication(self):
        """Retrieve the first available video publication."""
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.kind == rtc.TrackKind.KIND_VIDEO:
                    return publication
        return None

    async def _subscribe_and_capture_frame(self, publication):
        """Subscribe to the video publication and wait for a frame to be processed."""
        publication.set_subscribed(True)
        for _ in range(10):  # Wait up to 5 seconds
            if self.latest_video_frame:
                break
            await asyncio.sleep(0.5)

    def _unsubscribe_from_video(self, publication):
        """Unsubscribe from the video publication."""
        if publication:
            publication.set_subscribed(False)

def get_llm_model():
    """Get the LLM model based on environment settings."""
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if llm_provider == "ollama":
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        # Change model to one that supports function calling/tools
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        logger.info(f"Using Ollama LLM with model {ollama_model} at {ollama_url}")
        
        # Make sure the base URL is correct - we don't want to add /v1 if it's already there
        base_url = ollama_url
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
            
        try:
            # Use llama3 model which supports function calling
            return openai.LLM.with_ollama(
                model=ollama_model,
                base_url=base_url
            )
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM: {e}")
            logger.warning("Falling back to OpenAI LLM")
            # Fall back to OpenAI if Ollama initialization fails
            openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            return openai.LLM(model=openai_model)
    else:
        # Default to OpenAI
        # Use gpt-3.5-turbo if API key is not valid for gpt-4o
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        logger.info(f"Using OpenAI LLM with model {openai_model}")
        return openai.LLM(model=openai_model)

class TextModeAssistant:
    """A text-only assistant that works without TTS."""
    
    def __init__(self, room, llm_model, chat_ctx, fnc_ctx=None):
        self.room = room
        self.llm_model = llm_model
        self.chat_ctx = chat_ctx
        self.fnc_ctx = fnc_ctx
        
    def start(self, room):
        """Start listening for messages"""
        room.on("data_received", self._on_data_received)
        logger.info("Text-only assistant mode activated")
    
    async def _on_data_received(self, data, participant):
        """Handle incoming text messages"""
        try:
            if data.topic == "lk.chat":
                message = data.data.decode('utf-8')
                logger.info(f"Received message: {message}")
                
                # Add user message to chat context
                self.chat_ctx.append(role="user", text=message)
                
                # Generate response
                response = await self.llm_model.generate(
                    chat_ctx=self.chat_ctx,
                    fnc_ctx=self.fnc_ctx
                )
                
                # Process response
                response_text = response.text
                self.chat_ctx.append(role="assistant", text=response_text)
                
                # Send text response
                await self.room.local_participant.publish_data(
                    response_text.encode('utf-8'), 
                    topic="lk.chat"
                )
                logger.info(f"Sent response: {response_text}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

async def entrypoint(ctx: JobContext):
    """Main entry point for the voice assistant job."""
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "You are Linz, a voice and vision assistant created by Lindsey Web Solutions. "
                "You should use short and concise responses. If the user asks you to use their camera, use the capture_and_add_image function."
            ),
        )
        fnc_ctx = AssistantFnc(chat_ctx=initial_ctx)
        fnc_ctx.room = ctx.room

        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                asyncio.create_task(fnc_ctx.process_video_stream(track))

        # Get LLM model based on environment settings
        llm_model = get_llm_model()
        
        # Determine if we should use voice or text-only mode
        use_text_only = os.getenv("TEXT_ONLY_MODE", "false").lower() == "true"
        
        if use_text_only:
            logger.info("Starting in text-only mode (voice disabled)")
            assistant = TextModeAssistant(ctx.room, llm_model, initial_ctx, fnc_ctx)
            assistant.start(ctx.room)
            
            # Send welcome message
            await ctx.room.local_participant.publish_data(
                "Hey, I'm online! How can I assist you? (Text-only mode)".encode('utf-8'), 
                topic="lk.chat"
            )
        else:
            # Try to initialize voice assistant with fallback to text-only mode
            try:
                tts = openai.TTS(model="tts-1")
                
                assistant = VoiceAssistant(
                    vad=silero.VAD.load(),
                    stt=deepgram.STT(),
                    llm=llm_model,
                    tts=tts,
                    chat_ctx=initial_ctx,
                    fnc_ctx=fnc_ctx,
                )
                
                assistant.start(ctx.room)
                await asyncio.sleep(1)
                
                try:
                    await assistant.say("Hey, I'm online! How can I assist you?", allow_interruptions=True)
                    logger.info("Voice assistant started successfully")
                except Exception as e:
                    logger.error(f"Error in initial greeting: {e}")
                    logger.warning("Switching to text-only mode due to TTS error")
                    
                    # Fall back to text-only mode
                    text_assistant = TextModeAssistant(ctx.room, llm_model, initial_ctx, fnc_ctx)
                    text_assistant.start(ctx.room)
                    
                    # Send welcome message
                    await ctx.room.local_participant.publish_data(
                        "Hey, I'm online! How can I assist you? (Voice assistant failed, using text-only mode)".encode('utf-8'), 
                        topic="lk.chat"
                    )
            except Exception as e:
                logger.error(f"Error initializing voice assistant: {e}")
                logger.warning("Falling back to text-only mode")
                
                # Fall back to text-only mode
                text_assistant = TextModeAssistant(ctx.room, llm_model, initial_ctx, fnc_ctx)
                text_assistant.start(ctx.room)
                
                # Send welcome message
                await ctx.room.local_participant.publish_data(
                    "Hey, I'm online! How can I assist you? (Voice assistant failed, using text-only mode)".encode('utf-8'), 
                    topic="lk.chat"
                )

        while True:
            await asyncio.sleep(10)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
