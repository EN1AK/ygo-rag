## Context
The production workspace is /home/ubuntu/qqbot. Existing collection is allowlisted to 819151070 and permits up to eight historical examples per request.
## Decisions
Generate a description after 200 messages from three speakers; refresh after 100 new messages at most once per Beijing day. Balance eight anonymous examples and include only numeric aggregate statistics and previous description. Check hourly; persist daily retries, watermark, and a 120-second lease. Clear/collection stop invalidates jobs; descriptions expire in 30 days. Put the description in user data during generation. Reuse card download bytes in a typed translation result and attach them only to the first successful reply.
## Risks
Sparse examples may miss some habits; prompt requires conservative generalization. Model and network failures retain the last usable profile. Runtime verification uses synthetic messages and does not send QQ test messages.
