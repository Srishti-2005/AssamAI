import numpy as np
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.models.db_models import Agent, RoutingLog, Post, Comment
from app.config import settings

class FeedbackLearningService:
    """
    Closed-Loop Feedback & Dynamic Profile Learning.
    Tracks engagement rate:
        R = (Comments Received) / (Notifications Sent)
    Applies negative gradient shift away from unengaged post embeddings:
        v_new = normalize(v_old - eta * (1/N) * sum(v_unengaged))
    """

    def __init__(self):
        self.default_lr = settings.FEEDBACK_LEARNING_RATE
        self.min_unengaged = settings.FEEDBACK_NEGATIVE_THRESHOLD

    def get_agent_metrics(self, db: Session, agent_id: str) -> Dict[str, Any]:
        """
        Calculates engagement statistics for a specific agent.
        """
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            return {"error": "Agent not found"}

        # Notifications sent to this agent
        notifications = db.query(RoutingLog).filter(
            RoutingLog.target_agent_id == agent_id,
            RoutingLog.status.in_(["NOTIFIED", "ENGAGED", "IGNORED"])
        ).all()

        total_notified = len(notifications)
        total_engaged = sum(1 for log in notifications if log.status == "ENGAGED")
        total_ignored = sum(1 for log in notifications if log.status in ["IGNORED", "NOTIFIED"])

        engagement_rate = (total_engaged / total_notified) if total_notified > 0 else 0.0

        return {
            "agent_id": agent.id,
            "company_name": agent.company_name,
            "total_notifications_sent": total_notified,
            "total_engagements": total_engaged,
            "total_unengaged": total_ignored,
            "engagement_rate": round(engagement_rate, 4)
        }

    def apply_negative_gradient_shift(
        self,
        db: Session,
        agent_id: str,
        learning_rate: float = None,
        min_unengaged: int = None
    ) -> Dict[str, Any]:
        """
        Dynamically adjusts the agent's interest embedding away from topics
        they systematically ignored.
        """
        lr = learning_rate or self.default_lr
        threshold = min_unengaged or self.min_unengaged

        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            return {"success": False, "message": "Agent not found"}

        current_emb = agent.get_embedding()
        if not current_emb:
            return {"success": False, "message": "Agent has no existing embedding"}

        # Query unengaged logs
        unengaged_logs = db.query(RoutingLog).filter(
            RoutingLog.target_agent_id == agent_id,
            RoutingLog.status.in_(["IGNORED", "NOTIFIED"])
        ).order_by(RoutingLog.created_at.desc()).limit(20).all()

        if len(unengaged_logs) < threshold:
            return {
                "success": False,
                "message": f"Insufficient unengaged events ({len(unengaged_logs)}/{threshold}) to shift profile."
            }

        # Gather unengaged post embeddings
        unengaged_vectors: List[np.ndarray] = []
        for log in unengaged_logs:
            post = db.query(Post).filter(Post.id == log.post_id).first()
            if post and post.get_embedding():
                unengaged_vectors.append(np.array(post.get_embedding(), dtype=np.float32))

        if not unengaged_vectors:
            return {"success": False, "message": "No valid post embeddings found for unengaged logs."}

        # Mathematical Negative Gradient Shift:
        # v_agent_new = normalize(v_agent_old - eta * (1/N) * sum(v_unengaged))
        v_old = np.array(current_emb, dtype=np.float32)
        avg_unengaged = np.mean(unengaged_vectors, axis=0)

        shift = lr * avg_unengaged
        v_new = v_old - shift

        # L2 Re-normalization
        norm = np.linalg.norm(v_new)
        if norm > 0:
            v_new = v_new / norm

        # Save new embedding
        agent.set_embedding(v_new.tolist())

        # Mark unengaged logs as IGNORED (processed)
        for log in unengaged_logs:
            if log.status == "NOTIFIED":
                log.status = "IGNORED"

        db.commit()

        # Compute cosine shift delta
        cos_similarity_before_after = float(np.dot(v_old, v_new) / (np.linalg.norm(v_old) * np.linalg.norm(v_new)))

        return {
            "success": True,
            "agent_id": agent.id,
            "company_name": agent.company_name,
            "unengaged_samples_used": len(unengaged_vectors),
            "learning_rate": lr,
            "vector_drift_cosine": round(1.0 - cos_similarity_before_after, 6),
            "status": "PROFILE_UPDATED"
        }

feedback_service = FeedbackLearningService()
