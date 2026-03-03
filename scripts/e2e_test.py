"""
End-to-end test against a running instance.

Prerequisites:
  - The API must be running (uvicorn src.main:app --port 8000)
  - Azure services must be configured (.env)

Usage:
  python -m scripts.e2e_test [base_url]
"""

import json
import sys

import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
API = f"{BASE_URL}/api/v1"


def test_health():
    print("1. Health check...")
    r = requests.get(f"{API}/health")
    assert r.status_code == 200, f"Health failed: {r.status_code}"
    print(f"   OK: {r.json()}")


def test_create_cpg():
    print("\n2. Create a new CPG...")
    r = requests.post(
        f"{API}/chat",
        json={"message": "Create a clinical practice guideline for management of Type 2 Diabetes in elderly patients"},
    )
    assert r.status_code == 200, f"Create failed: {r.status_code} {r.text}"
    data = r.json()
    print(f"   Session: {data['session_id']}")
    print(f"   Response length: {len(data['response'])} chars")
    print(f"   Response preview: {data['response'][:200]}...")
    return data["session_id"]


def test_modify_cpg(session_id: str):
    print("\n3. Modify the CPG (change target population)...")
    r = requests.post(
        f"{API}/chat",
        json={
            "session_id": session_id,
            "message": "Change the target population to focus on patients over 75 years with renal impairment",
        },
    )
    assert r.status_code == 200, f"Modify failed: {r.status_code} {r.text}"
    data = r.json()
    print(f"   Response length: {len(data['response'])} chars")
    print(f"   Response preview: {data['response'][:200]}...")


def test_review_cpg(session_id: str):
    print("\n4. Review the CPG...")
    r = requests.post(
        f"{API}/chat",
        json={
            "session_id": session_id,
            "message": "Review the current guideline for completeness and clinical accuracy",
        },
    )
    assert r.status_code == 200, f"Review failed: {r.status_code} {r.text}"
    data = r.json()
    print(f"   Response length: {len(data['response'])} chars")
    print(f"   Response preview: {data['response'][:200]}...")


def test_general_question(session_id: str):
    print("\n5. Ask a general question...")
    r = requests.post(
        f"{API}/chat",
        json={
            "session_id": session_id,
            "message": "What are the key differences between SGLT2 inhibitors and DPP-4 inhibitors?",
        },
    )
    assert r.status_code == 200, f"Question failed: {r.status_code} {r.text}"
    data = r.json()
    print(f"   Response length: {len(data['response'])} chars")
    print(f"   Response preview: {data['response'][:200]}...")


def main():
    print(f"Running E2E tests against {API}\n{'='*60}")
    try:
        test_health()
        session_id = test_create_cpg()
        test_modify_cpg(session_id)
        test_review_cpg(session_id)
        test_general_question(session_id)
        print(f"\n{'='*60}")
        print("ALL E2E TESTS PASSED")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except requests.ConnectionError:
        print(f"\nCONNECTION ERROR: Is the API running at {BASE_URL}?")
        sys.exit(1)


if __name__ == "__main__":
    main()
