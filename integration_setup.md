# Integration Setup

To configure integrations, developers must first generate an API key from the developer settings page.

Sandbox endpoints should be used during development and testing.

Production endpoints should only be enabled after sandbox testing is completed successfully.

Webhook integrations require signature verification to ensure request authenticity.

If webhook deliveries fail, the platform retries automatically with exponential backoff.

Customers experiencing duplicate webhook deliveries should ensure their systems support idempotent processing.