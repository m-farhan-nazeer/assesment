# API Rate Limits

API rate limits depend on the subscription tier.

Basic plans are limited to 60 API requests per minute.

Pro plans are limited to 300 API requests per minute.

Enterprise customers have custom rate limits defined in their contract.

If a client exceeds the allowed rate, the API returns HTTP status code 429.

Clients should implement retry logic using exponential backoff when rate limits occur.