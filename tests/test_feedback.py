import pytest
import numpy as np
from app.database.session import Base, engine, SessionLocal
from app.models.db_models import Agent, Post, RoutingLog
from app.services.feedback import feedback_service
from app.services.embedding import embedding_service

@pytest.fixture
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

def test_closed_loop_engagement_metrics(db):
    agent = Agent(
        company_name="TestCorp",
        industry_category="Finance",
        webhook_url="http://test/webhook",
        interest_manifesto="General finance"
    )
    db.add(agent)
    db.commit()

    # Create dummy post
    post = Post(
        author_agent_id=agent.id,
        title="Finance update",
        content="Market trends"
    )
    db.add(post)
    db.commit()

    # Add 2 logs: 1 ENGAGED, 1 IGNORED
    log1 = RoutingLog(
        post_id=post.id,
        target_agent_id=agent.id,
        vector_score=0.85,
        rerank_score=0.85,
        status="ENGAGED"
    )
    log2 = RoutingLog(
        post_id=post.id,
        target_agent_id=agent.id,
        vector_score=0.80,
        rerank_score=0.80,
        status="IGNORED"
    )
    db.add_all([log1, log2])
    db.commit()

    metrics = feedback_service.get_agent_metrics(db, agent.id)
    assert metrics["total_notifications_sent"] == 2
    assert metrics["total_engagements"] == 1
    assert metrics["engagement_rate"] == 0.5

def test_negative_gradient_shift_calculation(db):
    initial_text = "Automotive engineering and spare parts"
    initial_emb = embedding_service.embed_text(initial_text)

    agent = Agent(
        company_name="AutoCorp",
        industry_category="Automotive",
        webhook_url="http://test/webhook",
        interest_manifesto=initial_text
    )
    agent.set_embedding(initial_emb)
    db.add(agent)
    db.commit()

    # Simulate 3 unengaged food delivery posts received by AutoCorp
    unengaged_posts = []
    for i in range(3):
        p = Post(
            author_agent_id=agent.id,
            title=f"Spam Food Delivery #{i}",
            content="Fast food delivery discount burger pizza coupon"
        )
        p_emb = embedding_service.embed_text(p.content)
        p.set_embedding(p_emb)
        db.add(p)
        db.commit()
        unengaged_posts.append(p)

        log = RoutingLog(
            post_id=p.id,
            target_agent_id=agent.id,
            vector_score=0.75,
            rerank_score=0.75,
            status="IGNORED"
        )
        db.add(log)
    db.commit()

    # Apply negative gradient shift
    res = feedback_service.apply_negative_gradient_shift(
        db=db,
        agent_id=agent.id,
        learning_rate=0.1,
        min_unengaged=3
    )

    assert res["success"] is True
    assert res["unengaged_samples_used"] == 3
    assert res["vector_drift_cosine"] > 0.0

    # Ensure updated vector is normalized
    updated_agent = db.query(Agent).filter(Agent.id == agent.id).first()
    new_emb = np.array(updated_agent.get_embedding(), dtype=np.float32)
    norm = np.linalg.norm(new_emb)
    assert np.isclose(norm, 1.0, atol=1e-4)
