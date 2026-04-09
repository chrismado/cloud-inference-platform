from __future__ import annotations

from fastapi.testclient import TestClient

from router.slo_router import app


def main() -> None:
    client = TestClient(app)

    print("Health:", client.get("/health").json())
    print("Text request:", client.post("/infer", json={"request_type": "text", "prompt": "hello"}).json())
    print("Video request:", client.post("/infer", json={"request_type": "video", "prompt": "orbit shot"}).json())


if __name__ == "__main__":
    main()
