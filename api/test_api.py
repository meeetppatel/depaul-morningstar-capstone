"""
Test script for GECS Classification API
Run after starting the server: uvicorn main:app --reload
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("=== Health Check ===")
    print(json.dumps(r.json(), indent=2))
    print()

def test_single():
    payload = {
        "company_id"  : "TEST001",
        "long_profile": "the company is an american packaged-food manufacturer producing frozen vegetables, meals and spices",
        "as_of_date"  : "2024",
        "total_revenue": 3000000000,
        "segments": [
            {
                "name"         : "Frozen and Vegetables",
                "description"  : "frozen and vegetables segment includes the green giant and le sueur brands",
                "revenue_share": 0.205,
                "is_largest"   : False,
            },
            {
                "name"         : "Meals",
                "description"  : "meals segment includes ortega, maple grove farms and cream of wheat brands",
                "revenue_share": 0.239,
                "is_largest"   : True,
            },
            {
                "name"         : "Spices and Flavor Solutions",
                "description"  : "spices segment includes dash, spice islands and weber brands",
                "revenue_share": 0.205,
                "is_largest"   : False,
            },
        ]
    }
    r = requests.post(f"{BASE_URL}/predict/full", json=payload)
    print("=== Single Company Prediction ===")
    data = r.json()
    print(f"Company: {data['company_id']}")
    print(f"Total latency: {data['total_latency_ms']}ms")
    for pred in data["predictions"]:
        print(f"\n  Segment: {pred['segment_name']}")
        print(f"    Industry   : {pred['industry_code']} (conf={pred['industry_confidence']:.3f})")
        print(f"    SubIndustry: {pred['subindustry_code']} (conf={pred['subindustry_confidence']:.3f})")
        print(f"    Route      : {pred['route']}")
        print(f"    Review?    : {pred['needs_review']}")
    print()

def test_batch():
    payload = {
        "companies": [
            {
                "company_id"  : "BATCH001",
                "long_profile": "semiconductor company producing wafers and substrates",
                "segments": [
                    {"name": "Substrates", "description": "high-performance compound semiconductor substrates", "revenue_share": 0.70},
                    {"name": "Raw Materials", "description": "sale of raw materials for substrate production", "revenue_share": 0.30},
                ]
            },
            {
                "company_id"  : "BATCH002",
                "long_profile": "financial institution providing banking services",
                "segments": [
                    {"name": "Commercial Banking", "description": "commercial and industrial loans and deposits", "revenue_share": 0.60},
                    {"name": "Wealth Management", "description": "investment advisory and wealth management services", "revenue_share": 0.40},
                ]
            },
        ]
    }
    r = requests.post(f"{BASE_URL}/predict/full", json=payload)
    print("=== Batch Prediction ===")
    data = r.json()
    print(f"Companies: {data['total_companies']}")
    print(f"Segments : {data['total_segments']}")
    print(f"Latency  : {data['total_latency_ms']}ms")
    print(f"Routes   : {data['routes_summary']}")
    print()

def test_industry_only():
    payload = {
        "company_id": "IND001",
        "long_profile": "energy company operating oil and gas exploration",
        "segments": [
            {"name": "Exploration", "description": "oil and gas exploration and production", "revenue_share": 1.0}
        ]
    }
    r = requests.post(f"{BASE_URL}/predict/industry", json=payload)
    print("=== Industry Only ===")
    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    test_health()
    test_single()
    test_batch()
    test_industry_only()
