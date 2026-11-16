import pytest
from app.database.session import Base, engine
from simulation.api_client import DirectApiClient

@pytest.fixture
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    return DirectApiClient()

def test_agent_registration_lifecycle(client):
    # Register agent
    resp = client.post("/api/v1/agents/register", json={
        "company_name": "PhonePe",
        "industry_category": "Payments & Fintech",
        "webhook_url": "https://agent.phonepe.com/v1/network-events",
        "interest_manifesto": "E-commerce and food delivery discounts"
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["company_name"] == "PhonePe"
    assert "webhook_secret" in data
    assert data["is_active"] is True

    # Duplicate registration fails
    dup_resp = client.post("/api/v1/agents/register", json={
        "company_name": "PhonePe",
        "industry_category": "Payments & Fintech",
        "webhook_url": "https://agent.phonepe.com/v1/network-events",
        "interest_manifesto": "E-commerce and food delivery discounts"
    })
    assert dup_resp.status_code == 400

    # List agents
    list_resp = client.get("/api/v1/agents")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

def test_post_creation_and_autonomous_comment(client):
    # Register Swiggy
    swiggy = client.post("/api/v1/agents/register", json={
        "company_name": "Swiggy",
        "industry_category": "Food Delivery",
        "webhook_url": "http://swiggy.internal/events",
        "interest_manifesto": "Food delivery deals"
    }).json()

    # Register PhonePe
    phonepe = client.post("/api/v1/agents/register", json={
        "company_name": "PhonePe",
        "industry_category": "Payments",
        "webhook_url": "http://phonepe.internal/events",
        "interest_manifesto": "Food delivery cashback and payments"
    }).json()

    # Publish post
    post_resp = client.post("/api/v1/posts", json={
        "author_agent_id": swiggy["id"],
        "title": "Diwali Food Offer",
        "content": "Flat 20% discount on festive meals above 400."
    })
    assert post_resp.status_code == 201
    post = post_resp.json()

    # Autonomous comment from PhonePe
    comment_resp = client.post(f"/api/v1/posts/{post['id']}/comments", json={
        "author_agent_id": phonepe["id"],
        "content": "PhonePe gives extra 50 cashback with UPI payment."
    })
    assert comment_resp.status_code == 201
    comment = comment_resp.json()
    assert comment["author_agent_id"] == phonepe["id"]

    # Verify post comments thread
    get_post_resp = client.get(f"/api/v1/posts/{post['id']}")
    assert get_post_resp.status_code == 200
    assert len(get_post_resp.json()["comments"]) == 1

    # Check that routing log status updated to ENGAGED
    logs_resp = client.get(f"/api/v1/routing/logs?post_id={post['id']}")
    logs = logs_resp.json()
    phonepe_log = next((l for l in logs if l["target_agent_id"] == phonepe["id"]), None)
    assert phonepe_log is not None
    assert phonepe_log["status"] == "ENGAGED"
