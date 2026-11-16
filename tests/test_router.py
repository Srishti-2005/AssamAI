import pytest
from app.database.session import Base, engine, SessionLocal
from app.models.db_models import Agent, Post, RoutingLog
from app.services.router import router
from app.services.embedding import embedding_service
import json

@pytest.fixture
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

def test_stage_1_metadata_guardrails(db):
    # Setup publisher
    swiggy = Agent(
        company_name="Swiggy",
        industry_category="Food Delivery",
        webhook_url="http://test/webhook",
        interest_manifesto="Food delivery offers"
    )
    swiggy.set_embedding(embedding_service.embed_text("Food delivery offers"))

    # Deactivated agent
    inactive_agent = Agent(
        company_name="Inactive Corp",
        industry_category="Retail",
        webhook_url="http://test/webhook",
        interest_manifesto="General retail",
        is_active=False
    )
    inactive_agent.set_embedding(embedding_service.embed_text("General retail"))

    # Agent with blocked industry
    blocked_agent = Agent(
        company_name="AntiFood Corp",
        industry_category="Manufacturing",
        webhook_url="http://test/webhook",
        interest_manifesto="Heavy industry manufacturing",
        blocked_industries=json.dumps(["Food Delivery"])
    )
    blocked_agent.set_embedding(embedding_service.embed_text("Heavy industry manufacturing"))

    db.add_all([swiggy, inactive_agent, blocked_agent])
    db.commit()

    post = Post(
        author_agent_id=swiggy.id,
        title="Food discount",
        content="50% off on all pizzas"
    )
    db.add(post)
    db.commit()

    qualified, logs = router.route_post(db, post)

    # Neither inactive nor blocked agent should qualify
    assert len(qualified) == 0

    # Inactive agent was filtered by SQL
    # Blocked agent logged as DROPPED_STAGE_1
    blocked_log = db.query(RoutingLog).filter(RoutingLog.target_agent_id == blocked_agent.id).first()
    assert blocked_log is not None
    assert blocked_log.status == "DROPPED_STAGE_1"

def test_stage_2_and_3_relevance_matching(db):
    # Register Swiggy
    swiggy = Agent(
        company_name="Swiggy",
        industry_category="Food Delivery",
        webhook_url="http://test/webhook",
        interest_manifesto="Food delivery festive discounts"
    )
    swiggy.set_embedding(embedding_service.embed_text("Food delivery festive discounts"))

    # Relevant partner: PhonePe
    phonepe = Agent(
        company_name="PhonePe",
        industry_category="Payments & Fintech",
        webhook_url="http://test/webhook",
        interest_manifesto="Interested in e-commerce promotions, food delivery festive discounts, card payment partnerships, merchant cashback"
    )
    phonepe.set_embedding(embedding_service.embed_text(
        f"{phonepe.company_name} {phonepe.industry_category} {phonepe.interest_manifesto}"
    ))

    # Irrelevant partner: Bajaj Auto
    bajaj = Agent(
        company_name="Bajaj Auto",
        industry_category="Automotive & Manufacturing",
        webhook_url="http://test/webhook",
        interest_manifesto="Automotive engineering, dealership supply chains, spare parts logistics, two-wheeler manufacturing"
    )
    bajaj.set_embedding(embedding_service.embed_text(
        f"{bajaj.company_name} {bajaj.industry_category} {bajaj.interest_manifesto}"
    ))

    db.add_all([swiggy, phonepe, bajaj])
    db.commit()

    post = Post(
        author_agent_id=swiggy.id,
        title="Diwali Food Delivery Discount",
        content="Swiggy offering flat 5% off on all delivery orders above 300 during Diwali week."
    )
    db.add(post)
    db.commit()

    qualified, logs = router.route_post(db, post)

    # PhonePe must qualify, Bajaj Auto must be dropped
    assert len(qualified) == 1
    assert qualified[0].id == phonepe.id

    phonepe_log = next(l for l in logs if l.target_agent_id == phonepe.id)
    assert phonepe_log.rerank_score >= 0.75
    assert phonepe_log.status == "NOTIFIED"

    bajaj_log = next(l for l in logs if l.target_agent_id == bajaj.id)
    assert bajaj_log.rerank_score < 0.75
    assert bajaj_log.status == "DROPPED_STAGE_3"
