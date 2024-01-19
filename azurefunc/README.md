# Notes

Azure functions by themselves are no good for long running tasks. Instead, use Azure Durable Functions which use blog storage and an orchestrator layer to provide a stateful API.

Calling a durable function by HTTP will return JSON containing URLs you can call to get status updates.

## Powershell

- install azure dev kit
- `py -m venv .venv`
- `.venv\scripts\activate` 
- 

## Local dev

Azure durable functions need blob storage to store state and optionally for input/output.

Install Azurite emulator to avoid the need for hosted blob storage.

