#!/usr/bin/env python3
"""End-to-end sandbox test for list follow-ups. Touches ONLY todo.sandbox — never the family lists.

Usage:  HASS_TOKEN=... python tests/sandbox_voice_chain.py [http://192.168.68.82:8123] [agent_id]
Creates a fresh chain: first-turn add + three follow-ups ("Yes, add", "Also", "Yes, add"),
then a show, asserts every item landed via the todo API, then removes them. Exit 1 on any miss.
Requires a `Sandbox` local_todo list (todo.sandbox) — create via Settings > Integrations > Local To-do.
"""
import json, os, sys, time, urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://192.168.68.82:8123").rstrip("/")
AGENT = sys.argv[2] if len(sys.argv) > 2 else "conversation.purellm_lm_studio_local"
TOK = os.environ.get("HASS_TOKEN") or sys.exit("set HASS_TOKEN")
H = {"Authorization": f"Bearer {TOK}", "Content-Type": "application/json"}
LIST = "todo.sandbox"
STAMP = str(int(time.time()))[-4:]
ITEMS = [f"sb bananas {STAMP}", f"sb oat milk {STAMP}", f"sb peanut butter {STAMP}", f"sb coffee filters {STAMP}"]
TURNS = [f"Add {ITEMS[0]} to the sandbox list.", f"Yes, add {ITEMS[1]}.", f"Also {ITEMS[2]}.", f"Yes, add {ITEMS[3]}."]


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers=H, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def items():
    d = post("/api/services/todo/get_items?return_response", {"entity_id": LIST})
    return [i["summary"] for i in d["service_response"][LIST]["items"]]


ok = True
cid = None
for t in TURNS:
    body = {"text": t, "agent_id": AGENT, "language": "en"}
    if cid:
        body["conversation_id"] = cid
    t0 = time.time()
    d = post("/api/conversation/process", body)
    cid = d.get("conversation_id", cid)
    speech = d["response"]["speech"]["plain"]["speech"].replace("\n", " | ")
    print(f"{time.time()-t0:5.2f}s | {t!r:45} -> {speech[:80]}")
    if "|" in speech or len(speech) > 120:
        print("   !! suspicious long / multi-line reply (loop?)"); ok = False

present = items()
for it in ITEMS:
    hit = it in present
    print(("  ok  " if hit else "  MISS") + " " + it)
    ok &= hit

for it in ITEMS:
    if it in present:
        post("/api/services/todo/remove_item", {"entity_id": LIST, "item": it})
print("cleanup done, sandbox now:", items())
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
