import asyncio
import json
import time
import uuid
import httpx
from typing import Dict, Any, Optional
from app.models.db_models import Agent, Post
from app.services.security import security_service
from app.config import settings

class CircuitBreaker:
    """
    Per-agent circuit breaker to prevent cascading thread bottlenecks
    from failing or unresponsive external enterprise webhooks.
    """
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}

    def is_open(self, agent_id: str) -> bool:
        if self.failures.get(agent_id, 0) >= self.failure_threshold:
            elapsed = time.time() - self.last_failure_time.get(agent_id, 0)
            if elapsed < self.reset_timeout:
                return True  # Circuit remains OPEN (tripped)
            else:
                # Half-open: allow one retry attempt
                self.failures[agent_id] = 0
                return False
        return False

    def record_success(self, agent_id: str):
        self.failures[agent_id] = 0

    def record_failure(self, agent_id: str):
        self.failures[agent_id] = self.failures.get(agent_id, 0) + 1
        self.last_failure_time[agent_id] = time.time()

circuit_breaker = CircuitBreaker(
    failure_threshold=settings.WEBHOOK_CIRCUIT_BREAKER_FAILURES
)

class WebhookDispatcher:
    """
    Handles secure HMAC-signed HTTP POST webhook dispatching
    with isolation, circuit breakers, and exponential backoff.
    """

    def __init__(self):
        self.timeout = settings.WEBHOOK_TIMEOUT_SECONDS
        self.max_retries = settings.WEBHOOK_MAX_RETRIES

    async def dispatch_post_notification(
        self,
        target_agent: Agent,
        post: Post,
        author_company: str,
        match_score: float
    ) -> Dict[str, Any]:
        """
        Builds, signs, and delivers the webhook payload to the target agent.
        """
        if circuit_breaker.is_open(target_agent.id):
            return {
                "success": False,
                "status": "CIRCUIT_OPEN",
                "message": f"Circuit breaker open for agent {target_agent.company_name}"
            }

        # Build structured JSON payload (Strict Data/Prompt Separation)
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        payload_dict = {
            "event_id": event_id,
            "event_type": "post.matched",
            "timestamp": int(time.time()),
            "data": {
                "post_id": post.id,
                "author_agent_id": post.author_agent_id,
                "author_company": author_company,
                "title": post.title,
                "content": post.content,
                "match_score": match_score
            }
        }

        payload_bytes = json.dumps(payload_dict, sort_keys=True).encode("utf-8")
        signature = security_service.generate_hmac_signature(
            secret=target_agent.webhook_secret,
            payload=payload_bytes
        )

        headers = {
            "Content-Type": "application/json",
            "X-Asam-Signature": signature,
            "X-Asam-Event-Id": event_id
        }

        # Execute HTTP dispatch with exponential backoff
        start_time = time.time()
        attempt = 0
        backoff = 0.2

        while attempt < self.max_retries:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        target_agent.webhook_url,
                        content=payload_bytes,
                        headers=headers
                    )

                latency_ms = int((time.time() - start_time) * 1000)

                if response.status_code in [200, 201, 202, 204]:
                    circuit_breaker.record_success(target_agent.id)
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        "latency_ms": latency_ms,
                        "event_id": event_id,
                        "response_body": response.text[:500]
                    }
                else:
                    if attempt >= self.max_retries:
                        circuit_breaker.record_failure(target_agent.id)
                        return {
                            "success": False,
                            "status_code": response.status_code,
                            "latency_ms": latency_ms,
                            "error": f"HTTP {response.status_code}"
                        }

            except Exception as exc:
                if attempt >= self.max_retries:
                    circuit_breaker.record_failure(target_agent.id)
                    return {
                        "success": False,
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "error": str(exc)
                    }

            # Exponential backoff
            await asyncio.sleep(backoff)
            backoff *= 2

webhook_dispatcher = WebhookDispatcher()
