## ADDED Requirements
### Requirement: Automatic bounded style description
The service SHALL produce a style description only for the enabled group, after 200 valid messages from at least three members, and refresh after 100 new messages at most once per Beijing date. Each request SHALL contain at most eight historical samples, no speaker identifiers, and aggregate numeric statistics.
#### Scenario: Insufficient samples
- **WHEN** fewer than 200 messages or three members are available initially
- **THEN** no summary model request is issued
#### Scenario: Eligible refresh
- **WHEN** 100 new messages are available on a later Beijing date
- **THEN** a bounded job is reserved and a successful description is used in subsequent replies
### Requirement: Lifecycle and inspection
The service SHALL persist leases and watermarks, discard results after collection is disabled or cleared, delete profiles on clear or 30-day expiry, and expose the current profile through @神人 群聊风格.
#### Scenario: Clear during generation
- **WHEN** an administrator clears collection while a summary is running
- **THEN** the old result cannot recreate the deleted profile
