# Talk_MCP_for_Gemini_CLI
An MCP for Gemini CLI to allow voice 

### IMPORTANT ### 
This was writtenf or Windows WSL2 on Ubuntu. 

It will not likely work on other systems without some changes. This is expected. I will try to get it working on MacOS next. 

### There is nothign to do here ###
You will add this to your Gemini settings.json file as such: 

    "talk": {
      "command": "/home/<homedir>/Talk_MCP_for_Gemini_CLI/.env/bin/python",
      "args": [
        "/home/<homedir>/Talk_MCP_for_Gemini_CLI/talk.py"
      ]
    },

As you see you need to setup a virtual environment and install the requirements.txt file in this MCP directory for it work.

fastmcp         2.11.3
openai-whisper  20250625
PyAudio         0.2.14

### MacOS 
Add 
brew install portaudio
brew install ffmpeg
python3 -m pip install -r requirements.txt

** Make sure your open-ai whisper is using the venv not the global whisper install **

#### tailf 
tail -f /tmp/talk_mcp.log

This will show you wtf is going on with the MCP while it works :D 