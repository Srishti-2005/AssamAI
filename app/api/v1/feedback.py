from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.services.feedback import feedback_service

router = APIRouter(prefix="/feedback", tags=["Closed-Loop Feedback"])

@router.get("/metrics/{agent_id}")
def get_agent_feedback_metrics(agent_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Returns engagement rate R = Comments Received / Notifications Sent for an agent.
    """
    metrics = feedback_service.get_agent_metrics(db, agent_id)
    if "error" in metrics:
        raise HTTPException(status_code=404, detail=metrics["error"])
    return metrics

@router.post("/update-profiles/{agent_id}")
def trigger_dynamic_profile_update(
    agent_id: str,
    learning_rate: Optional[float] = Query(None, ge=0.001, le=0.5),
    min_unengaged: Optional[int] = Query(None, ge=1, le=50),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Applies the mathematical negative gradient shift to stale or unengaged agent manifestos.
    """
    result = feedback_service.apply_negative_gradient_shift(
        db=db,
        agent_id=agent_id,
        learning_rate=learning_rate,
        min_unengaged=min_unengaged
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result
