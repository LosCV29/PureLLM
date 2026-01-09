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

## Custom Wake Words (Voice PE)

Want to use a custom wake word like "Hey Jarvis" instead of the default? See our detailed guide:

**[Custom Wake Word Setup for Voice PE](docs/custom-wake-word-voice-pe.md)**

The guide covers:
- Modifying Voice PE firmware via ESPHome
- Training your own wake word models
- Using the openWakeWord add-on
- Best practices for wake word selection

## Difference from PolyVoice

PureLLM is a fork of [PolyVoice](https://github.com/LosCV29/polyvoice) with a key architectural difference:

- **PolyVoice**: Hybrid approach - can intercept native HA intents and route specific devices/intents to LLM
- **PureLLM**: Pure LLM approach - all commands go through LLM tools, no native intent interception

Choose PureLLM if you want consistent LLM-powered responses for everything.

## License

MIT License - See LICENSE file for details.
