## Why
Random group chat has examples but no persistent style description; translated card replies omit the submitted image.
## What Changes
Add daily bounded style synthesis and use the stored description in group replies. Return the original card image with its translation.
## Capabilities
### New Capabilities
- group-style-summary: aggregate style synthesis, lifecycle and user inspection.
### Modified Capabilities
- card-image-translation: return the same submitted image with the translated text.
## Impact
Server kami-man storage/scheduler/reply composition and ygo-rag style endpoint. Existing group allowlist and raw-example bounds remain.
