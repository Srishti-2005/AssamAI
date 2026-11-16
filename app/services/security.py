import hmac
import hashlib
import re
from typing import Tuple

# Common prompt injection and adversarial jailbreak patterns
SUSPICIOUS_PROMPT_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"system\s+override",
    r"output\s+(your\s+)?(system\s+prompt|confidential|secret|api\s*key)",
    r"reveal\s+(your\s+)?(secret|password|private\s+key)",
    r"format\s+drive|execute\s+command|rm\s+-rf",
    r"bypass\s+(safety|guardrail|filters)",
]

class SecurityService:
    """
    Handles cryptographic webhook signing (HMAC-SHA256) and
    content guardrail sanitization to prevent prompt injection hijacking.
    """

    @staticmethod
    def generate_hmac_signature(secret: str, payload: bytes) -> str:
        """
        Signs raw JSON payload with HMAC-SHA256 using the agent's shared secret.
        Returns header format: 'sha256=<hex_digest>'
        """
        signature = hmac.new(
            key=secret.encode("utf-8"),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    @staticmethod
    def verify_hmac_signature(secret: str, payload: bytes, signature_header: str) -> bool:
        """
        Constant-time verification of HMAC-SHA256 signature to protect against timing attacks.
        """
        if not signature_header or not signature_header.startswith("sha256="):
            return False
        
        expected_sig = signature_header.split("sha256=")[1].strip()
        actual_sig = hmac.new(
            key=secret.encode("utf-8"),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(actual_sig, expected_sig)

    @staticmethod
    def inspect_content_for_prompt_injection(text: str) -> Tuple[bool, str]:
        """
        Guardrail sanitizer checking incoming posts for adversarial prompt injection.
        Returns (is_safe: bool, reason: str).
        """
        lowered = text.lower()
        for pattern in SUSPICIOUS_PROMPT_PATTERNS:
            if re.search(pattern, lowered):
                return False, f"Potential prompt injection detected matching pattern: {pattern}"

        return True, "Content passed guardrail inspection."

security_service = SecurityService()
