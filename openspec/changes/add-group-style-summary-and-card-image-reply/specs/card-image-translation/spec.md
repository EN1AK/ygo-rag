## ADDED Requirements
### Requirement: Original image accompanies translation
The bot SHALL return the same downloaded image bytes with the first successful translation reply and SHALL NOT repeat the image in later text chunks.
#### Scenario: Long translated text
- **WHEN** a translated card requires multiple messages
- **THEN** only the first contains the original image and all text chunks are preserved
#### Scenario: Recognition failure
- **WHEN** card recognition fails
- **THEN** the existing error and refund behavior applies and no image reply is generated
