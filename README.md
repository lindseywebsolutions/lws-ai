# LiveKit Assistant

First, create a virtual environment, update pip, and install the required packages:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

You need to set up the following environment variables:

```
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
```

Then, run the assistant:

```
$ python3 assistant.py download-files
$ python3 assistant.py start
```

## Using Ollama

If you want to use Ollama with the DeepSeek-R1 model:

1. Install Ollama following the instructions at https://ollama.com/
2. Pull the DeepSeek model: `ollama pull deepseek-r1`
3. Make sure Ollama is running: Ollama runs as a service on port 11434
4. Set `LLM_PROVIDER=ollama` in your .env file

## Viewing

Finally, you can load the [hosted playground](https://agents-playground.livekit.io/) and connect it.