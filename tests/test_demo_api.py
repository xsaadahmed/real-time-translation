from fastapi.testclient import TestClient

from rtt.api.demo_server import app


def test_simulated_socket_finishes_without_microphone():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["type"] == "update"
            socket.send_json({"type": "stop"})
            assert socket.receive_json()["type"] == "progress"
            assert socket.receive_json()["type"] == "final"


def test_production_api_routes_simulated_mode_without_live_models(monkeypatch):
    monkeypatch.setenv("RTT_FALLBACK_ONLY", "1")
    from rtt.api.production_server import app as production_app

    with TestClient(production_app) as client:
        with client.websocket_connect("/ws?mode=simulated") as socket:
            assert socket.receive_json()["type"] == "update"
            socket.send_json({"type": "stop"})
            assert socket.receive_json()["type"] == "progress"
            assert socket.receive_json()["type"] == "final"
