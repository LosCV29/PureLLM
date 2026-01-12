# PureLLM

Pure LLM Voice Assistant for Home Assistant - all voice commands flow through the LLM pipeline.

## Overview

PureLLM is a streamlined voice assistant integration that routes **all** voice commands through an LLM (Large Language Model) for intelligent processing. Unlike hybrid approaches, PureLLM doesn't intercept native Home Assistant intents - every command goes through the LLM tool-calling pipeline.

## Features

- **Pure LLM Pipeline**: All voice commands processed by your chosen LLM
- **Multi-Provider Support**: OpenAI, Anthropic, Google, Groq, OpenRouter, Azure, Ollama, LM Studio
- **Smart Home Control**: Lights, switches, covers, thermostats, locks, fans via LLM tools
- **Music Assistant Integration**: Full music control through LLM
- **External Services**: Weather, news, sports, restaurants, places, Wikipedia
- **Camera Integration**: Works with HA Video Vision for camera queries
- **Calendar Integration**: Query your Home Assistant calendars

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Click the three dots menu → Custom repositories
3. Add `https://github.com/LosCV29/purellm` as an Integration
4. Search for "PureLLM" and install
5. Restart Home Assistant
6. Add the integration via Settings → Devices & Services → Add Integration → PureLLM

### Manual Installation

1. Download the `custom_components/purellm` folder
2. Copy it to your Home Assistant `config/custom_components/` directory
3. Restart Home Assistant
4. Add the integration via Settings → Devices & Services

## Configuration

1. Select your LLM provider
2. Enter API credentials
3. Configure features (weather, music, etc.)
4. Set up device aliases for voice control
5. Configure music room mappings if using Music Assistant

## Difference from PolyVoice

PureLLM is a fork of [PolyVoice](https://github.com/LosCV29/polyvoice) with a key architectural difference:

- **PolyVoice**: Hybrid approach - can intercept native HA intents and route specific devices/intents to LLM
- **PureLLM**: Pure LLM approach - all commands go through LLM tools, no native intent interception

Choose PureLLM if you want consistent LLM-powered responses for everything.

## Wake Word Setup with Porcupine

PureLLM works with Home Assistant's voice pipeline system. To activate PureLLM hands-free, you'll need to set up wake word detection. This guide covers setting up **Porcupine** with a custom wake word.

### Architecture Overview

```
Audio Input → Wake Word Detection (Porcupine) → STT (Speech-to-Text) → PureLLM → TTS (Response)
```

PureLLM receives text input from HA's conversation platform. Wake word detection happens upstream in the voice pipeline.

### Option A: openWakeWord (Easiest - Recommended for Beginners)

Home Assistant's built-in openWakeWord add-on is the simplest option:

1. Go to **Settings → Add-ons → Add-on Store**
2. Search for and install **openWakeWord**
3. Start the add-on
4. Go to **Settings → Devices & Services** and configure the Wyoming integration when discovered
5. Go to **Settings → Voice assistants** → Create/edit an assistant
6. Set **Conversation agent** to `PureLLM`
7. Click the three-dot menu → **Add streaming wake word** → Select `openwakeword`
8. Choose a wake word (e.g., "ok nabu", "hey jarvis")

### Option B: Porcupine with Custom Wake Word

For a **custom wake word** (e.g., your own branded phrase), use Porcupine:

#### Step 1: Get a Picovoice Access Key

1. Sign up at [Picovoice Console](https://console.picovoice.ai/) (free tier available)
2. Navigate to **AccessKey** in your account
3. Copy your access key - you'll need this for configuration

#### Step 2: Create Your Custom Wake Word

1. In Picovoice Console, go to **Porcupine** section
2. Select your target **language** (English, Spanish, French, German, etc.)
3. Select your target **platform** (Raspberry Pi, Linux, etc.)
4. Enter your custom wake word phrase (e.g., "Hey Assistant", "Computer", "Jarvis")
5. Click **Train** - the model trains in seconds
6. Download the `.ppn` file (this is your custom wake word model)

> **Note:** Free tier `.ppn` files expire after 30 days. You can regenerate them anytime from the console.

#### Step 3: Install Wyoming-Porcupine Add-on

For Home Assistant OS/Supervised:

1. Add this custom repository to your add-on store:
   ```
   https://github.com/rhasspy/hassio-addons
   ```
2. Install the **Porcupine1** add-on (or search for community Porcupine v3 add-ons)
3. Configure the add-on with your access key and `.ppn` file path

#### Step 4: Configure Home Assistant Voice Pipeline

1. Go to **Settings → Devices & Services**
2. Add the **Wyoming** integration if not auto-discovered
3. Point it to your Porcupine add-on (usually `localhost:10400`)
4. Go to **Settings → Voice assistants**
5. Create or edit a voice assistant:
   - **Name**: Your wake word name
   - **Language**: Match your wake word language
   - **Conversation agent**: `PureLLM`
   - **Speech-to-text**: Your STT engine (Whisper, Google, etc.)
   - **Text-to-speech**: Your TTS engine (Piper, etc.)
   - **Wake word**: Select your Porcupine wake word

### Option C: PorcupinePipeline (Advanced - Standalone Device)

For running wake word detection on a separate device (e.g., Raspberry Pi with mic):

1. Clone [PorcupinePipeline](https://github.com/slackr31337/PorcupinePipeline):
   ```bash
   git clone https://github.com/slackr31337/PorcupinePipeline.git
   cd PorcupinePipeline
   ```

2. Install dependencies:
   ```bash
   pip install pvporcupine pyaudio websocket-client
   ```

3. Get a long-lived access token from Home Assistant:
   - Go to your HA profile → **Security** → **Long-lived access tokens** → Create token

4. Copy your custom `.ppn` file to the device

5. Run the pipeline:
   ```bash
   export ACCESS_KEY='your-picovoice-key'
   export TOKEN='your-ha-long-lived-token'
   export SERVER='homeassistant.local'
   export PIPELINE='PureLLM'  # Name of your voice assistant
   export KEYWORD_PATHS='/path/to/your-wake-word.ppn'
   export AUDIO_DEVICE=1  # Your microphone device index

   python3 voice_pipeline.py --server $SERVER --pipeline "$PIPELINE" --follow-up
   ```

### Hardware Recommendations

For always-listening wake word detection:

- **M5Stack ATOM Echo** - Affordable, works with ESPHome
- **Raspberry Pi + USB Microphone** - Flexible, good for PorcupinePipeline
- **ReSpeaker** - Quality microphone arrays with far-field pickup

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Wake word not detected | Check microphone sensitivity, reduce background noise |
| Too many false positives | Train a longer/more unique wake word phrase |
| `.ppn` file expired | Regenerate in Picovoice Console (free) |
| "Device limit reached" | Picovoice limits devices; use same access key or upgrade plan |
| Pipeline not triggering | Verify Wyoming integration is connected and pipeline is configured |

### Recommended Pipeline Configuration

For the best experience with PureLLM:

| Component | Recommended Option |
|-----------|-------------------|
| Wake Word | Porcupine or openWakeWord |
| STT | Faster Whisper (local) or Whisper (OpenAI API) |
| Conversation | PureLLM |
| TTS | Piper (local) or your preferred cloud TTS |

## License

MIT License - See LICENSE file for details.
