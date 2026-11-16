from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database.session import get_db
from app.models.db_models import RoutingLog
from app.models.schemas import RoutingLogResponse

router = APIRouter(prefix="/routing", tags=["Routing Telemetry"])

@router.get("/logs", response_model=List[RoutingLogResponse])
def get_routing_logs(
    post_id: Optional[str] = Query(None),
    target_agent_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    Retrieves routing telemetry logs for auditability, feedback tracking, and model monitoring.
    """
    query = db.query(RoutingLog)
    if post_id:
        query = query.filter(RoutingLog.post_id == post_id)
    if target_agent_id:
        query = query.filter(RoutingLog.target_agent_id == target_agent_id)
    if status:
        query = query.filter(RoutingLog.status == status)

    return query.order_by(RoutingLog.created_at.desc()).limit(limit).all()

@router.get("/stats")
def get_routing_statistics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Provides platform-level routing funnel statistics.
    """
    total_evaluations = db.query(RoutingLog).count()
    status_counts = db.query(
        RoutingLog.status, func.count(RoutingLog.id)
    ).group_by(RoutingLog.status).all()

    breakdown = {status: count for status, count in status_counts}

    avg_vector = db.query(func.avg(RoutingLog.vector_score)).scalar() or 0.0
    avg_rerank = db.query(func.avg(RoutingLog.rerank_score)).scalar() or 0.0

    return {
        "total_evaluations": total_evaluations,
        "status_breakdown": breakdown,
        "average_vector_score": round(avg_vector, 4),
        "average_rerank_score": round(avg_rerank, 4)
    }
