import json
import logging
from typing import Dict, Any, Optional
from app.services.security import security_service

logger = logging.getLogger("MockAgent")

class MockEnterpriseAgent:
    """
    Simulates an external enterprise autonomous agent (e.g. PhonePe, Bajaj Auto, HDFC Bank).
    Runs local decision logic, verifies platform HMAC signatures, and autonomously
    calls the comment API to execute cross-company partnerships.
    """

    def __init__(self, agent_id: str, company_name: str, webhook_secret: str, api_client=None):
        self.agent_id = agent_id
        self.company_name = company_name
        self.webhook_secret = webhook_secret
        self.api_client = api_client
        self.received_events = []
        self.posted_comments = []

    def handle_incoming_webhook(self, payload_bytes: bytes, signature_header: str) -> Dict[str, Any]:
        """
        1. Verifies HMAC-SHA256 signature from AsamAI platform
        2. Unpacks structured JSON data (Strict Data/Prompt Separation)
        3. Evaluates event and autonomously triggers response if valuable
        """
        # Cryptographic Verification
        is_valid = security_service.verify_hmac_signature(
            secret=self.webhook_secret,
            payload=payload_bytes,
            signature_header=signature_header
        )
        if not is_valid:
            logger.error(f"[{self.company_name}] Signature verification FAILED! Rejecting payload.")
            return {"status": "REJECTED_INVALID_HMAC", "authenticated": False}

        payload = json.loads(payload_bytes.decode("utf-8"))
        self.received_events.append(payload)
        event_data = payload.get("data", {})
        post_id = event_data.get("post_id")
        title = event_data.get("title", "")
        content = event_data.get("content", "")
        author_company = event_data.get("author_company", "")
        match_score = event_data.get("match_score", 0.0)

        # Autonomous Decision Making Engine
        response_comment = self._evaluate_and_generate_response(author_company, title, content)

        if response_comment and self.api_client and post_id:
            # Autonomously call comments API
            comment_resp = self.api_client.post(
                f"/api/v1/posts/{post_id}/comments",
                json={
                    "author_agent_id": self.agent_id,
                    "content": response_comment
                }
            )
            self.posted_comments.append(response_comment)
            return {
                "status": "ENGAGED_AUTONOMOUS_RESPONSE_POSTED",
                "authenticated": True,
                "comment": response_comment,
                "api_status": comment_resp.status_code
            }

        return {
            "status": "RECEIVED_NO_ACTION",
            "authenticated": True
        }

    def _evaluate_and_generate_response(self, author_company: str, title: str, content: str) -> Optional[str]:
        """
        Autonomous rule-based & LLM-style decision logic tailored per enterprise.
        """
        combined = f"{title} {content}".lower()

        if self.company_name == "PhonePe":
            if "diwali" in combined or "food" in combined or "delivery" in combined or "swiggy" in combined:
                return f"PhonePe will provide extra ₹50 cashback when users pay via PhonePe UPI for {author_company} Diwali orders."
            elif "ecommerce" in combined or "shopping" in combined:
                return f"PhonePe offers 10% instant checkout discount up to ₹150 for all {author_company} transactions."

        elif self.company_name == "HDFC Bank":
            if "food" in combined or "dining" in combined or "swiggy" in combined:
                return f"HDFC Bank offers an exclusive 15% cashback for Swiggy HDFC Credit Card holders during Diwali festive week."

        elif self.company_name == "Bajaj Auto":
            # Automotive agent has no interest in food delivery
            return None

        return None
