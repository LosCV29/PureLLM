# Custom Wake Word Setup for Home Assistant Voice PE

This guide explains how to add custom wake words to your Home Assistant Voice PE speaker so you can use personalized activation phrases like "Hey Jarvis" or your own custom name.

## Overview

The Voice PE uses **microWakeWord** for on-device wake word detection. There are two approaches to add custom wake words:

1. **ESPHome Modification** (Recommended) - Modify the Voice PE firmware directly
2. **openWakeWord Add-on** - Use server-side wake word detection

## Option 1: ESPHome Firmware Modification (Recommended)

This method modifies the Voice PE's ESPHome configuration to add custom microWakeWord models.

### Prerequisites

- Home Assistant with ESPHome add-on installed
- Voice PE device set up and working
- A trained microWakeWord model (.tflite + .json files)

### Step 1: Take Control of Device in ESPHome

1. Go to **Settings** → **Add-ons** → **ESPHome**
2. Open the ESPHome dashboard
3. Find your Voice PE device
4. Click the three dots menu → **Take Control**

This creates a local copy of the device configuration you can modify.

### Step 2: Get a Custom Wake Word Model

You have several options:

#### Option A: Use Pre-trained Community Models

Check the [esphome/micro-wake-word-models](https://github.com/esphome/micro-wake-word-models) repository for available models.

#### Option B: Train Your Own Model

Use one of these training tools:

1. **Docker Trainer** (Easiest): [microWakeWord-Custom-Trainer](https://github.com/stujenn/microWakeWord-Custom-Trainer)
   ```bash
   git clone https://github.com/stujenn/microWakeWord-Custom-Trainer.git
   cd microWakeWord-Custom-Trainer
   docker-compose up
   ```
   The web interface will guide you through training.

2. **Apple Silicon Trainer**: [microWakeWord-Trainer-AppleSilicon](https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon)
   For M1/M2/M3 Macs with GPU acceleration.

3. **Google Colab Notebook**: [basic_training_notebook](https://github.com/OHF-Voice/micro-wake-word)
   For advanced users who want full control.

### Step 3: Host Your Model Files

Your trained model produces two files:
- `your_wake_word.tflite` - The model file
- `your_wake_word.json` - Model metadata

Host these files so ESPHome can download them:

**Option A: GitHub Repository**
1. Create a GitHub repository
2. Upload both files
3. Get the raw file URLs

**Option B: Local Web Server**
1. Place files in your HA `/config/www/` folder
2. Access via `http://your-ha-ip:8123/local/your_wake_word.tflite`

### Step 4: Modify ESPHome Configuration

Edit your Voice PE configuration in ESPHome:

```yaml
# Add to your voice-pe.yaml configuration

substitutions:
  name: voice-pe
  friendly_name: Voice PE

# Include the base Voice PE package
packages:
  esphome.voice-pe: github://esphome/firmware/voice-pe/esp32-s3-voice-kit-v1.yaml@main

# Override the micro_wake_word configuration
micro_wake_word:
  models:
    # Add your custom wake word
    - model: https://your-host.com/path/to/your_wake_word.json
      id: custom_wake_word

    # Keep one default if you want (optional)
    - model: hey_jarvis  # Built-in model
      id: hey_jarvis_model

  # Remove default wake words you don't want (required due to space limits)
  # !remove is used to remove models from the base package
```

**Important**: The ESP32-S3 has limited flash space. You typically need to remove some default wake words to add custom ones.

### Step 5: Remove Unwanted Default Wake Words

To remove default wake words and make room for custom ones:

```yaml
micro_wake_word:
  models:
    # Your custom model
    - model: https://your-host.com/path/to/your_wake_word.json
      id: custom_wake_word

  # Remove these from the base package
  on_wake_word_detected: !remove
```

Or selectively remove specific models by overriding the entire `models` list.

### Step 6: Compile and Upload

1. In ESPHome dashboard, click **Install** on your Voice PE
2. Choose **Wirelessly** to update over-the-air
3. Wait for the compilation and upload to complete

### Step 7: Configure in Home Assistant

1. Go to **Settings** → **Voice Assistants**
2. Edit your voice assistant configuration
3. Select your new wake word from the dropdown

## Option 2: openWakeWord Add-on (Alternative)

This method runs wake word detection on your Home Assistant server instead of the device.

### Prerequisites

- Home Assistant OS or Supervised installation
- Microphone input streaming capability

### Step 1: Install openWakeWord Add-on

1. Go to **Settings** → **Add-ons**
2. Click **Add-on Store**
3. Search for "openWakeWord"
4. Install and start the add-on

### Step 2: Configure Wyoming Integration

1. Go to **Settings** → **Devices & Services**
2. Add the **Wyoming** integration
3. Point it to the openWakeWord add-on

### Step 3: Train Custom Wake Word

Use the [openWakeWord training notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb):

1. Open the Google Colab notebook
2. Enter your desired wake word (3-4 syllables works best)
3. Run all cells (takes ~1 hour)
4. Download the generated `.tflite` file

### Step 4: Add to Home Assistant

1. Access your HA server via Samba/SSH
2. Create folder: `/share/openwakeword/`
3. Copy your `.tflite` file there
4. Restart the openWakeWord add-on
5. Configure your voice assistant to use the new wake word

### Alternative: open-voice-pe Firmware

For full openWakeWord support on the device itself, consider the [open-voice-pe](https://github.com/mike-nott/open-voice-pe) community firmware:

1. Replaces the stock Voice PE firmware
2. Adds native openWakeWord support
3. Enables remote TTS playback to other speakers
4. Requires ESPHome 2025.3.0+

## Wake Word Best Practices

### Choosing a Good Wake Word

- **Use 3-4 syllables**: "Hey Jarvis", "OK Computer", "Hey Cassandra"
- **Avoid common words**: Don't use words that appear often in TV/music
- **Unique sounds**: Words with distinct consonant sounds work better
- **Test thoroughly**: Check for false activations during normal conversation

### Tested Working Wake Words

Community-tested wake words that work well:
- "Hey Jarvis"
- "Hey Mycroft"
- "OK Nabu" (default)
- "Hey Cassandra"
- "Alexa" (if you're not using Amazon devices)

### Troubleshooting

**Wake word not detected:**
- Speak clearly at normal volume
- Ensure you're within 2-3 meters of the device
- Check ESPHome logs for detection events

**Too many false activations:**
- Train with more negative samples
- Try a more unique wake word phrase
- Adjust the detection threshold in ESPHome config

**Device runs out of memory:**
- Remove unused default wake words
- Use only 1-2 wake words total

## Integration with PureLLM

Once your custom wake word is configured, PureLLM works automatically:

1. Voice PE detects your custom wake word
2. Speech is captured and sent to HA's speech-to-text
3. Text is routed to PureLLM's conversation agent
4. PureLLM processes the command through your configured LLM
5. Response is spoken back via text-to-speech

No additional PureLLM configuration is needed - wake word detection happens at the device/HA level before PureLLM receives the command.

## Resources

- [microWakeWord GitHub](https://github.com/OHF-Voice/micro-wake-word) - Training framework
- [ESPHome micro_wake_word docs](https://esphome.io/components/micro_wake_word/) - Component documentation
- [Pre-trained models](https://github.com/esphome/micro-wake-word-models) - Ready-to-use models
- [open-voice-pe](https://github.com/mike-nott/open-voice-pe) - Alternative firmware
- [HA Voice Control docs](https://www.home-assistant.io/voice_control/about_wake_word/) - Official documentation
